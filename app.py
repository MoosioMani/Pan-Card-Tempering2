import os
import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity
from PIL import Image

st.set_page_config(page_title="PAN Card Tampering Detection", layout="centered")
st.title("PAN Card Tampering Detection")
st.markdown(
    "Upload two images - the **original** PAN card and the **suspect** image. "
    "The app compares them using Structural Similarity Index (SSIM) and highlights "
    "regions where they differ."
)

# File upload
orig_file = st.file_uploader(
    "Upload original PAN card image", type=["jpg", "jpeg", "png"], key="orig"
)
suspect_file = st.file_uploader(
    "Upload suspect PAN card image", type=["jpg", "jpeg", "png"], key="suspect"
)

if orig_file and suspect_file:
    # Read images
    orig_img = Image.open(orig_file).convert("RGB")
    suspect_img = Image.open(suspect_file).convert("RGB")

    # Resize to same size
    max_width = max(orig_img.width, suspect_img.width)
    max_height = max(orig_img.height, suspect_img.height)
    orig_resized = orig_img.resize((max_width, max_height))
    suspect_resized = suspect_img.resize((max_width, max_height))

    # Convert to grayscale
    orig_gray = cv2.cvtColor(np.array(orig_resized), cv2.COLOR_RGB2GRAY)
    suspect_gray = cv2.cvtColor(np.array(suspect_resized), cv2.COLOR_RGB2GRAY)

    # Compute SSIM
    score, diff = structural_similarity(orig_gray, suspect_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold and find contours
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw rectangles on copies
    orig_draw = np.array(orig_resized).copy()
    suspect_draw = np.array(suspect_resized).copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(orig_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(suspect_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display results
    st.write(f"**SSIM Score:** {score*100:.2f} % (100% = identical)")
    st.subheader("Original image with differences highlighted")
    st.image(orig_draw, use_column_width=True)
    st.subheader("Suspect image with differences highlighted")
    st.image(suspect_draw, use_column_width=True)
    st.subheader("Difference heatmap")
    st.image(diff, use_column_width=True)
