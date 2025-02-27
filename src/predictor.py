from keras.models import load_model
from PIL import Image
import numpy as np

CLASS_NAMES = ["Animals", "Anime", "ExpertiseInvoices"]

def preprocess_image(image_path, target_size=(150, 150)):
    """
    Loads an image, resizes it, and formats it for model input.
    """
    img = Image.open(image_path)
    
    # Convert 4-channel RGBA images to RGB
    if img.mode == "RGBA":
        img = img.convert("RGB")
    # Convert grayscale (1-channel) images to RGB
    elif img.mode == "L":
        img = img.convert("RGB")

    # Resize and convert to numpy array
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize

    # Reshape to match the model's expected input format (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=0)

    return img


def predict_image_class(model_path, image_path):
    """
    Makes a prediction using a trained model for the given image file.
    """
    model = load_model(model_path)
    img = preprocess_image(image_path)

    # Predict with the model
    prediction = model.predict(img)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    if confidence < 0.5:
        predicted_class_name = "Unknown"
    else:
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
    return predicted_class_index, predicted_class_name, confidence
