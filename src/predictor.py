from keras.models import load_model
from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(150, 150)):
    img = Image.open(image_path).resize(target_size)
    img = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img

def predict_image_class(model_path, image_path):
    model = load_model(model_path)
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class
