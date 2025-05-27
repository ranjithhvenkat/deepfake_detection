# cnnvit_app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from utils import load_model, transform_image
from model import CNNViT 
st.set_page_config(page_title="Fake vs Real Detector")
st.title("🕵️ Fake vs Real Image Classifier")

# Load model
@st.cache_resource
def get_model():
    return load_model(CNNViT,"cnnvit_model.pth")

model = get_model()
class_names = ['Fake', 'Real']

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Predict
    if st.button("Predict"):
        input_tensor = transform_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        label = class_names[pred.item()]
        st.markdown(f"### 🔍 Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence.item() * 100:.2f}%")
