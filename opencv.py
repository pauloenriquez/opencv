import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import subprocess

def install_required_packages():
    packages = ['streamlit', 'numpy', 'opencv-python', 'Pillow']

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} not found. Installing...")
            subprocess.run(["pip", "install", package])


install_required_packages()
# Create a simple Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user to choose an image file
uploaded = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])

# Check if the user selected a file
if not uploaded:
    print("No file selected. Exiting.")
    exit()

# Load the selected image
img = cv2.imread(uploaded)
if img is None:
    print("Error: Unable to load the image.")
    exit()

# Rest of your code remains unchanged
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

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
