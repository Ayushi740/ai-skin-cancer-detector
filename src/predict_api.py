import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "models/skin_cancer_model.h5"
IMAGE_SIZE = (224, 224)

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# CLASS LABELS (7 classes from HAM10000)
CLASS_NAMES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanocytic Nevus",
    "Melanoma",
    "Vascular Lesions"
]

def predict(image_path):
    img = Image.open(image_path).resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    return CLASS_NAMES[class_index], confidence

if __name__ == "__main__":
    path = input("Enter image path: ")

    if not os.path.exists(path):
        print("❌ Image not found. Check your path.")
    else:
        label, conf = predict(path)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {conf:.2f}")
