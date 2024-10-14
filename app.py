import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("Image Conversion Web App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a format suitable for OpenCV
    img_array = np.array(image.convert('RGB'))

    st.sidebar.header("Choose an Operation")
    operation = st.sidebar.selectbox("Operation", ["Grayscale", "Flip", "Rotate", "Scale", "Translate", 
                                                   "Reflect", "Binary Threshold", "Inverse Binary", 
                                                   "Adaptive Threshold", "Sobel", "Canny"])

    # Operation: Grayscale
    if operation == "Grayscale":
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(gray_image, caption="Grayscale Image", use_column_width=True)
        result_image = gray_image

    # Operation: Flip (flip horizontally or vertically)
    elif operation == "Flip":
        flip_option = st.sidebar.radio("Flip Direction", ("Horizontal", "Vertical"))
        if flip_option == "Horizontal":
            flipped_image = cv2.flip(img_array, 1)
        else:
            flipped_image = cv2.flip(img_array, 0)
        st.image(flipped_image, caption=f"Flipped {flip_option}", use_column_width=True)
        result_image = flipped_image

    # Operation: Rotate
    elif operation == "Rotate":
        angle = st.sidebar.slider("Rotation Angle", 0, 360, 90)
        (h, w) = img_array.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(img_array, rotation_matrix, (w, h))
        st.image(rotated_image, caption=f"Rotated by {angle} degrees", use_column_width=True)
        result_image = rotated_image


    if operation in ["Grayscale", "Flip", "Rotate"]:
        if st.button("Download Image"):
            if len(result_image.shape) == 2:  # Grayscale image
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)

            download_image = Image.fromarray(result_image)
            download_image.save("processed_image.png")
            st.success("Image saved! You can download it from the file system.")
