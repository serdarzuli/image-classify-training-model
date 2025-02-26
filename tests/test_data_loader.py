import pytest
import numpy as np
from src.data_loader import load_images_and_labels
from src.model_trainer import create_model
from src.predictor import predict_image_class


def test_load_images_and_labels():
    images, labels = load_images_and_labels("data/raw/test")
    assert isinstance(images, np.ndarray), "Images should be a NumPy array"
    assert isinstance(labels, np.ndarray), "Labels should be a NumPy array"
    assert images.shape[0] == labels.shape[0], "Number of images and labels should match"
