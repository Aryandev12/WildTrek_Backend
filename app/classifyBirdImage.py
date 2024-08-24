import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import json


BIRD_IMAGE_DATA = None
with open("WildTrek_Backend/app/static/bird_image.json", "r") as f:
    BIRD_IMAGE_DATA = json.load(f)

def classify_bird_image_set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
device = classify_bird_image_set_device()
bird_image_checkpoint = torch.load("WildTrek_Backend/app/models/bird_image_checkpoint.pth.tar", weights_only=False, map_location=device)
bird_image_model = torch.load("WildTrek_Backend/app/models/bird_image.pth", map_location=device, weights_only=False)

mean = bird_image_checkpoint['mean']
std = bird_image_checkpoint['std']

image_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))  # image = (image - mean) / std
])


def classify_bird_image(image_path,model=bird_image_model, image_transforms=image_transforms, data=BIRD_IMAGE_DATA):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)

    # Compute probabilities
    probabilities = F.softmax(output, dim=1)
    max_prob, predicted = torch.max(probabilities.data, 1)
    predicted = predicted.item()


    result_not_found = {
        "scientific_name": "Could not identify",
        "common_name": "Could not identify",
        "description": "Could not identify",
        "habitat": "Could not identify",
        "endangered": "Could not identify",
        "dangerous": "Could not identify",
        "poisonous": "Could not identify",
        "venomous": "Could not identify",
        "probability": 0
    }
    result = None
    for scientificName in data.keys():
        if scientificName == predicted.lower():
            result = {
                "scientific_name": scientificName,
                "common_name": data[scientificName]["commonName"],
                "description": data[scientificName]["description"],
                "habitat": data[scientificName]["habitat"],
                "endangered": str(data[scientificName]["isEndangered"]),
                "dangerous": str(data[scientificName]["isDangerous"]),
                "poisonous": str(data[scientificName]["poisonous"]),
                "venomous": str(data[scientificName]["venomous"]),
                "probability": max_prob.item() * 100
            }
    try:
        if result['probability'] >=20:
            return result
    except:
        return result_not_found

