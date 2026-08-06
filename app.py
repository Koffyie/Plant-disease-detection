import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

#Loading model and class labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.keras')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Invert dict: {0: 'Apple___Black_rot', 1: ...}
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

#App UI
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detector")
st.write("Upload a photo of a plant leaf and get a disease prediction.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    #Preprocess to match training pipeline
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)[0]

    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx] * 100
    predicted_class = idx_to_class[top_idx].replace("___", " — ").replace("_", " ")

    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.1f}%")

    #Reveal top 3 predictions
    st.write("---")
    st.write("**Top 3 predictions:**")
    top3_idx = np.argsort(predictions)[-3:][::-1]
    for idx in top3_idx:
        cls_name = idx_to_class[idx].replace("___", " — ").replace("_", " ")
        st.write(f"- {cls_name}: {predictions[idx]*100:.1f}%")