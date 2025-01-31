import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

# Import necessary functions
from model import load_model, process_image, predict, overlay_images

# Streamlit Page Configuration
st.set_page_config(page_title="Brain Tumor Segmentation",page_icon='🧠', layout="wide")

# Custom CSS for better aesthetics
st.markdown(
    """
    <style>
        body {
            background-color: #fafafa;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            padding: 10px 0;
        }
        .note {
            text-align: center;
            font-size: 16px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        .image-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        img {
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Note
st.markdown('<div class="title">🧠 Brain Tumor Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="note">Upload an MRI image and visualize the segmentation results.</div>', unsafe_allow_html=True)

# File Uploader
uploaded_image = st.file_uploader("📤 Upload a brain MRI image", type=['png', 'jpg', 'jpeg', 'tif'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Convert Streamlit image to NumPy array
    image_np = np.array(image)

    # Ensure it has 3 channels (RGB)
    if len(image_np.shape) == 2:  # Grayscale image case
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Convert to OpenCV BGR format
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("interface stremlit\\best_tversky_model.pth", device)

    # Process Image
    model_input = process_image(image_np)

    # Generate Prediction
    predicted_mask_np = predict(model, model_input)

    # Overlay Mask on MRI Image
    overlay = overlay_images(model_input, predicted_mask_np)

    # Convert NumPy arrays to images for Streamlit display
    model_input_pil = Image.fromarray((model_input.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    predicted_mask_pil = Image.fromarray((predicted_mask_np * 255).astype(np.uint8))
    overlay_pil = Image.fromarray(overlay)

    # Display images
    st.markdown("### 🖼️ Segmentation Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
    with col2:
        st.image(predicted_mask_pil, caption="🧠 Predicted Mask", use_container_width=True)
        
    with col3:
        st.image(overlay_pil, caption="🔍 Overlayed Image", use_container_width=True)
