import pytest
import numpy as np
from src.data_loader import load_images_and_labels
from src.model_trainer import create_model
from src.predictor import predict_image_class


@pytest.mark.asyncio
def test_predict_image_class():
    model_path = "data/models/final_model.keras"
    image_path = "data/raw/test/sample_image.jpg"
    predicted_class = predict_image_class(model_path, image_path)
    assert isinstance(predicted_class, np.ndarray), "Prediction result should be an array"
    assert predicted_class.shape == (1,), "Prediction should return a single class"
