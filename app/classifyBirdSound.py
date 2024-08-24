import torchaudio
import torch
import json
from torchvision.models.mobilenetv2 import mobilenet_v2
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


BIRD_SOUND_DATA = None
with open("WildTrek_Backend/app/static/bird_audio.json", "r") as f:
    BIRD_SOUND_DATA = json.load(f)

def classify_bird_sound_set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
device = classify_bird_sound_set_device()


class BirdSoundMobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BirdSoundMobileNet, self).__init__()
        
        
        self.mobilenet = mobilenet_v2(pretrained=pretrained)
        
        
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        
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

        return mel_spec.unsqueeze(0)  
    
bird_sound_preprocessor = AudioPreprocessor()
bird_sound_checkpoint = torch.load("WildTrek_Backend/app/models/bird_audio.pth", map_location=device)
bird_sound_idx_to_class = {v: k for k, v in bird_sound_checkpoint['class_to_idx'].items()}
bird_sound_num_classes = len(bird_sound_idx_to_class)
bird_sound_model = BirdSoundMobileNet(num_classes=bird_sound_num_classes)
bird_sound_model.load_state_dict(bird_sound_checkpoint['model_state_dict'])
bird_sound_model.to(device)

def predict_bird_species(model, audio_path, preprocessor, device, idx_to_class):
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

def classify_bird_sound(audio_path, model=bird_sound_model , preprocessor=bird_sound_preprocessor, data=BIRD_SOUND_DATA, idx_to_class=bird_sound_idx_to_class, device=device):
    predicted_species, confidence = predict_bird_species(model, audio_path, preprocessor, device, idx_to_class)

    result_not_found = {
        "scientific_name": "Could not identify",
        "common_name": "Could not identify",
        "description": "Could not identify",
        "habitat": "Could not identify",
        "endangered": "Could not identify",
        "dangerous": "Could not identify",
        "poisonous": "Could not identify",
        "venomous": "Could not identify",
        "probability": confidence * 100
    }
    result = None
    for scientificName in data.keys():
        if scientificName == predicted_species:
            result = {
                "scientific_name": scientificName,
                "common_name": data[scientificName]["commonName"],
                "description": data[scientificName]["description"],
                "habitat": data[scientificName]["habitat"],
                "endangered": str(data[scientificName]["isEndangered"]),
                "dangerous": str(data[scientificName]["isDangerous"]),
                "poisonous": str(data[scientificName]["poisonous"]),
                "venomous": str(data[scientificName]["venomous"]),
                "probability": confidence * 100
            }
    try:
        if result['probability'] >=20:
            return result
    except:
        return result_not_found