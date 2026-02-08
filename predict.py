import torch
import numpy as np
import tensorflow as tf
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path

# ==============================
# Paths
# ==============================
MODEL_PATH = r"D:\plant_disease_detection\plant_model.pth"

initial_path = input("Enter path")

# Example test image folder (choose one image at a time)
TEST_IMAGE = Path(initial_path)  # replace with your image

# ==============================
# Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Transform
# ==============================
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# Model
# ==============================
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==============================
# Prediction Function
# ==============================

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    classes = ["healthy", "infected"]
    return classes[predicted.item()]

# ==============================
# Run Prediction
# ==============================
result = predict_image(TEST_IMAGE)
print(f"The plant is: {result}")
