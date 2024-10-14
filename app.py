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
        
    # Operation: Scale
    elif operation == "Scale":
        scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0)
        width = int(img_array.shape[1] * scale_factor)
        height = int(img_array.shape[0] * scale_factor)
        scaled_image = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)
        st.image(scaled_image, caption=f"Scaled Image (Factor: {scale_factor})", use_column_width=True)
        result_image = scaled_image
        
    # Operation: Translate
    elif operation == "Translate":
        tx = st.sidebar.slider("Translation in X (pixels)", -100, 100, 0)
        ty = st.sidebar.slider("Translation in Y (pixels)", -100, 100, 0)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(img_array, translation_matrix, (img_array.shape[1], img_array.shape[0]))
        st.image(translated_image, caption=f"Translated Image (TX: {tx}, TY: {ty})", use_column_width=True)
        result_image = translated_image
    
    # Operation: Reflect
    elif operation == "Reflect":
        reflect_option = st.sidebar.radio("Reflect Direction", ("Horizontal", "Vertical"))
        if reflect_option == "Horizontal":
            reflected_image = cv2.flip(img_array, 1)
        else:
            reflected_image = cv2.flip(img_array, 0)
        st.image(reflected_image, caption=f"Reflected {reflect_option}", use_column_width=True)
        result_image = reflected_image
        
    # Operation: Binary Threshold
    elif operation == "Binary Threshold":
        threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 128)
        _, binary_image = cv2.threshold(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), threshold_value, 255, cv2.THRESH_BINARY)
        st.image(binary_image, caption="Binary Threshold Image", use_column_width=True)
        result_image = binary_image
        
    # Operation: Inverse Binary Threshold
    elif operation == "Inverse Binary":
        threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 128)
        _, inverse_binary_image = cv2.threshold(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), threshold_value, 255, cv2.THRESH_BINARY_INV)
        st.image(inverse_binary_image, caption="Inverse Binary Threshold Image", use_column_width=True)
        result_image = inverse_binary_image
        
    # Operation: Adaptive Threshold
    elif operation == "Adaptive Threshold":
        block_size = st.sidebar.slider("Block Size (Odd Number)", 3, 21, 11, step=2)
        adaptive_image = cv2.adaptiveThreshold(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
        st.image(adaptive_image, caption="Adaptive Threshold Image", use_column_width=True)
        result_image = adaptive_image
    
    # Operation: Sobel (Edge Detection)
    elif operation == "Sobel":
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        sobel_combined = np.uint8(sobel_combined)
        st.image(sobel_combined, caption="Sobel Edge Detection", use_column_width=True)
        result_image = sobel_combined

        
    # Operation: Canny Edge Detection
    elif operation == "Canny":
        threshold1 = st.sidebar.slider("Lower Threshold", 0, 255, 100)
        threshold2 = st.sidebar.slider("Upper Threshold", 0, 255, 200)
        canny_image = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), threshold1, threshold2)
        st.image(canny_image, caption="Canny Edge Detection", use_column_width=True)
        result_image = canny_image

    # Download button to save the processed image
    if st.button("Download Image"):
        if len(result_image.shape) == 2:  # Grayscale image or binary/edge images
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
        download_image = Image.fromarray(result_image)
        download_image.save("processed_image.png")
        st.success("Image saved! You can download it from the file system.")