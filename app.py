import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
# Replace the load_my_model function with this:
@st.cache_resource
def load_my_model():
    model = torch.load('plant_model.pth', map_location=torch.device('cpu'))
    model.eval()
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
    img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0)
    
    # Predict
    if st.button('Analyze'):
        predictions = model.predict(img_array)
        result = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        st.success(f"Prediction: {result} ({confidence:.2f}%)")


