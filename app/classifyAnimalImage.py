import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import json

# Load animal image data
ANIMAL_IMAGE_DATA = None
with open("app/static/animal_image.json", "r") as f:
    ANIMAL_IMAGE_DATA = json.load(f)

# Function to set the device (GPU or CPU)
def classify_animal_image_set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Set the device
device = classify_animal_image_set_device()

# Load the model and checkpoint
animal_image_checkpoint = torch.load("app/models/animal_image_checkpoint.pth.tar", map_location=device, weights_only=False)
animal_image_model = torch.load("app/models/animal_image.pth", map_location=device, weights_only=False)
animal_image_model = animal_image_model.to(device)

# Get mean and std from the checkpoint
mean = animal_image_checkpoint['mean']
std = animal_image_checkpoint['std']

# Define the image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Function to classify an animal image
def classify_animal_image(image_path, model=animal_image_model, image_transforms=image_transforms, data=ANIMAL_IMAGE_DATA):
    model = model.eval()

    # Load and transform the image
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0).to(device)  # Move image tensor to the same device as the model

    # Forward pass through the model
    output = model(image)

    # Compute probabilities
    probabilities = F.softmax(output, dim=1)
    max_prob, predicted = torch.max(probabilities.data, 1)
    predicted = predicted.item()
    predicted = list(data.keys())[predicted]

    # Define the result_not_found structure
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

    # Initialize result
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

    # Return result if probability is above 20%, else return result_not_found
    try:
        if result and result['probability'] >= 20:
            return result
    except:
        pass

    return result_not_found
