import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
# Replace the load_my_model function with this:
@st.cache_resource
def load_my_model():
    # Load the file
    checkpoint = torch.load('plant_model.pth', map_location=torch.device('cpu'))
    
    # If it's a dictionary (state_dict), you need the model class defined first
    # For now, let's assume it's the full model object
    if isinstance(checkpoint, dict):
        st.error("The .pth file is a state_dict. You must define the model architecture in app.py before loading.")
        return None
    
    model = checkpoint
    model.eval() # This sets it to inference mode
    return model



model = load_my_model()

# Define your class names (must match your model's output)
CLASS_NAMES = ['healthy', 'infected'] 

st.title("Crop Disease Detector")
st.write("Upload a leaf image to see the diagnosis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    # Preprocessing (adjust size based on your model's input)
   img = image.resize((224, 224)) 
img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0) # Change to (Batch, Channel, H, W)

# 2. Predict using PyTorch
if st.button('Analyze'):
    with torch.no_grad(): # Disable gradient calculation for speed
        outputs = model(img_tensor)
        # Get the index of the highest value
        _, predicted_idx = torch.max(outputs, 1)
        
    result = CLASS_NAMES[predicted_idx.item()]
    
    # Optional: Calculate confidence (Softmax)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence = probabilities[0][predicted_idx].item() * 100
    
    st.success(f"Prediction: {result} ({confidence:.2f}%)")



