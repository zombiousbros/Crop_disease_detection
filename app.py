import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
# Replace the load_my_model function with this:

@st.cache_resource
def load_my_model():
   model = models.resnet50(pretrained=False)
   num_features = model.fc.in_features
   model.fc = torch.nn.Linear(num_features, 2)
   model.load_state_dict(torch.load(plant_model.pth))
   model = model.to(device)
   model.eval() sets it to inference mode
return model



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






