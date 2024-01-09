import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import subprocess

def install_required_packages():
    packages = ['streamlit', 'numpy', 'opencv-python', 'Pillow']

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            st.warning(f"{package} not found. Installing...")
            subprocess.run(["pip", "install", package])

# Install required packages
install_required_packages()

# Streamlit app
st.title("Face and Eye Detection App")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Convert the image to OpenCV format
    img_np = np.array(image)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Face and eye detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the processed image
    st.image(img, caption="Processed Image.", use_column_width=True)

    # Optionally display coordinates or other information
    st.write(f"Number of faces detected: {len(faces)}")
