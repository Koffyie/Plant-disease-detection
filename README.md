# 🌿 Plant Disease Detection

A deep learning model that classifies plant leaf diseases from images, built with transfer learning on MobileNetV2. Trained on the PlantVillage dataset (38 classes across multiple crop species).

**🔗 Live demo:** [pladide.streamlit.app](https://pladide.streamlit.app)

## Problem

Crop diseases are a major cause of yield loss for smallholder farmers, and early detection often requires expert knowledge that isn't always accessible. This project explores whether a lightweight, deployable model can classify plant diseases from a single leaf photo — accurately enough to be a useful first-pass diagnostic tool.

## Dataset

- **Source:** [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes:** 38 (healthy and diseased leaves across multiple crop species)
- **Split:** 80% training / 20% validation
- **Preprocessing:** Images resized to 224×224, normalized, augmented (rotation, zoom, horizontal flip) to improve generalization

## Approach

- **Base model:** MobileNetV2, pretrained on ImageNet, used as a frozen feature extractor
- **Custom head:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → Dense(38, softmax)
- **Training:** Two-phase approach —
  1. Trained the classification head with the base frozen
  2. Fine-tuned the last 30 layers of MobileNetV2 at a low learning rate (1e-5)
- **Why MobileNetV2:** Chosen for its balance of accuracy and efficiency — small enough to deploy easily, while transfer learning from ImageNet gives strong performance without training from scratch

## Results

| Metric | Value |
|---|---|
| Training accuracy | 94.51% |
| Validation accuracy | 96.13% |
| Training loss | 0.1623 |
| Validation loss | 0.1155 |

![Training vs Validation Accuracy](accuracy_plot.png)

## Demo

The model is deployed as an interactive Streamlit app — upload a leaf photo and get a prediction with confidence score and top-3 alternatives.

## Tech stack

- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- Streamlit (deployment)
- PIL, NumPy

## Running locally

\`\`\`bash
git clone https://github.com/Koffyie/Plant-disease-detection.git
cd Plant-disease-detection
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Project structure

\`\`\`
plant-disease-detection/
├── app.py                          # Streamlit app
├── requirements.txt
├── plant_disease_model.keras       # Trained model
├── class_indices.json              # Class label mapping
├── accuracy_plot.png               # Training/validation accuracy plot
├── notebook/
│   └── plant_disease_detection.ipynb   # Training notebook
└── README.md
\`\`\`

## Author

Kofoworola Adekunle — [GitHub](https://github.com/Koffyie)