import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import os
from model import load_model, process_image, predict, overlay_images

# Streamlit Page Configuration
st.set_page_config(page_title="Brain Tumor Segmentation", page_icon='🧠', layout="wide")

# **Dark Mode & Custom Styling**
st.markdown(
    """
    <style>
    :root {
        --primary: #2b2b2b;
        --secondary: #4a4e69;
        --text: #ffffff;
    }
    
    body {
        background-color: var(--primary);
        color: var(--text);
    }
    
    .title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 10px 0;
        margin-bottom: 30px;
    }
    
    .note {
        text-align: center;
        font-size: 16px;
        color: #b0b0b0;
        margin-bottom: 30px;
    }

    .section-header {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 40px 0;
    }

    .image-caption {
        text-align: center;
        font-weight: bold;
        margin: 15px 0;
        font-size: 18px;
        color: #e0e0e0;
    }

    .centered-gif {
        display: flex;
        justify-content: center;
        margin: 30px 0;
    }

    /* Dark mode for uploader */
    .stFileUploader > div > div {
        border: 2px dashed var(--secondary) !important;
        background: rgba(74, 78, 105, 0.1) !important;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #e0e0e0;
        margin-top: 50px;
        padding: 20px;
        border-top: 2px solid var(--secondary);
    }

    /* Spinner color */
    .stSpinner > div > div {
        border-top-color: var(--secondary) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# **Title**
st.markdown('<div class="title">🧠 Brain Tumor Segmentation</div>', unsafe_allow_html=True)

# **Subtitle**
st.markdown('<div class="note">Upload an MRI image and visualize the segmentation results</div>', unsafe_allow_html=True)

# **Upload MRI Scan Section**
st.markdown("### 📤 Upload MRI Scan for Segmentation")
uploaded_image = st.file_uploader("Upload an MRI image", type=['png', 'jpg', 'jpeg', 'tif'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # Convert image to numpy array
    image_np = np.array(image)

    # Handle grayscale images
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Convert to BGR format
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # **Processing with Spinner**
    with st.spinner('🔍 Analyzing MRI scan...'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model("best_tversky_model.pth", device)
        model_input = process_image(image_np)
        predicted_mask_np = predict(model, model_input)
        overlay = overlay_images(model_input, predicted_mask_np)

        # Convert results to PIL images
        predicted_mask_pil = Image.fromarray((predicted_mask_np * 255).astype(np.uint8))
        overlay_pil = Image.fromarray(overlay)

    # **Segmentation Results**
    st.markdown("### 🖼️ Segmentation Results")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="image-caption">📷 Original Scan</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="image-caption">🧠 Tumor Detection</div>', unsafe_allow_html=True)
        st.image(predicted_mask_pil, use_container_width=True)

    with col3:
        st.markdown('<div class="image-caption">🔍 Tumor Overlay</div>', unsafe_allow_html=True)
        st.image(overlay_pil, use_container_width=True)

# **📥 Footer: Download Sample MRI Scans**
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
st.markdown("### 📥 Download Sample MRI Scans", unsafe_allow_html=True)
st.markdown("Download these MRI scans, then upload them above for segmentation.")

# **Load Sample MRI Scans for Individual Download with Dropdown**
sample_dir = "mri_scans"

if os.path.exists(sample_dir):
    sample_files = [file for file in os.listdir(sample_dir) if file.endswith((".png", ".jpg", ".jpeg", "tif"))]

    if sample_files:
        selected_sample = st.selectbox("🔍 Choose an MRI scan to preview:", ["Select a sample..."] + sample_files)

        if selected_sample != "Select a sample...":
            file_path = os.path.join(sample_dir, selected_sample)
            image_sample = Image.open(file_path)

            # Get natural size of image
            width, height = image_sample.size

            # Center the image using columns
            co1, co2, co3 = st.columns(3)

            with co2:
                st.image(image_sample, caption=f"📷 Preview: {selected_sample}", width=width)

            # Center the download button below the image
            with co2:
                with open(file_path, "rb") as f:
                    st.download_button(label=f"📥 Download {selected_sample}", data=f, file_name=selected_sample, mime="image/png")


    else:
        st.warning("No sample MRI scans found.")
else:
    st.error("The sample directory does not exist. Please add MRI scans to `mri_scans/`.")