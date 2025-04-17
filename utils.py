# src/utils.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("../trained_model/model.h5")

# Class label mapping (must match dataset folder names)
labels_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=']

def preprocess_drawing(path_points):
    canvas = np.ones((256, 256), dtype=np.uint8) * 255
    for i in range(1, len(path_points)):
        cv2.line(canvas, path_points[i - 1], path_points[i], 0, 8)
    img = cv2.resize(canvas, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)
    return img

def predict_symbol(image):
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    return labels_map[class_index]

