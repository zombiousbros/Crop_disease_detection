import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models



@st.cache_resource
def load_my_model():
    # 1. Initialize the architecture (THIS defines 'model')
    model = models.resnet50(weights=None)
    
    # 2. Modify the final layer to match YOUR number of classes (e.g., 3)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    # 3. Load the weights from your file
    checkpoint = torch.load('plant_model.pth', map_location='cpu')
    
    # 4. Fill the model with weights

    model.eval()
    return model

# 5. Call the function to actually create the variable for the rest of the script
model = load_my_model()




# Define your class names (must match your model's output)
CLASS_NAMES = ['healthy', 'infected'] 

st.title("Crop Disease Detector")
st.write("Upload a leaf image to see the diagnosis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_container_width=True)
    
    # 2. Preprocessing for PyTorch (Line 40 starts here)
    img = image.resize((224, 224)) 
    img_array = np.array(img).astype(np.float32) / 255.0
    # Convert to Tensor and change shape to (Batch, Channel, Height, Width)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # 3. Predict using PyTorch
    if st.button('Analyze'):
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
        
        result = CLASS_NAMES[predicted_idx.item()]
        st.success(f"Prediction: {result}")














