import torchaudio
import torch
import json
from torchvision.models.mobilenetv2 import mobilenet_v2
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class AnimalSoundMobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(AnimalSoundMobileNet, self).__init__()
        
        # Load a pre-trained MobileNetV2 model
        self.mobilenet = mobilenet_v2(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace the last fully connected layer to match the number of bird sound classes
        num_features = self.mobilenet.last_channel
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )
        
    def forward(self, x):
        x = self.mobilenet.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.mobilenet.classifier(x)
        return x

class AudioPreprocessor:
    def __init__(self, sample_rate=32000, n_mels=128, n_fft=1024, hop_length=512, duration=5, spec_width=313):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.spec_width = spec_width
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        desired_length = self.duration * self.sample_rate
        if waveform.shape[1] < desired_length:
            waveform = torch.nn.functional.pad(waveform, (0, desired_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :desired_length]

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        if mel_spec.shape[2] < self.spec_width:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, self.spec_width - mel_spec.shape[2]))
        else:
            mel_spec = mel_spec[:, :, :self.spec_width]

        return mel_spec.unsqueeze(0)  # Add batch dimension


def predict_animal_species(model, audio_path, preprocessor, device, idx_to_class):
    model.eval()
    audio_tensor = preprocessor.preprocess(audio_path)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        output = model(audio_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    predicted_species = idx_to_class[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_species, confidence