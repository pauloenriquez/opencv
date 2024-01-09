import subprocess

def install_required_packages():
    packages = ['streamlit', 'numpy', 'opencv-python']

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} not found. Installing...")
            subprocess.run(["pip", "install", package])

def detect_faces_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame

def main():
    # Install required packages
    install_required_packages()

    import streamlit as st
    import cv2
    import numpy as np

    st.title("Face and Eye Detection with OpenCV and Streamlit")
    st.sidebar.header("Settings")

    # Load the Haar cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while st.sidebar.button("Start Detection"):
        ret, frame = cap.read()

        if not ret:
            st.warning("Unable to fetch the webcam feed.")
            break

        frame = detect_faces_and_eyes(frame)

        # Display the frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

    # Release the webcam and close the Streamlit app
    cap.release()

if __name__ == "__main__":
    main()
