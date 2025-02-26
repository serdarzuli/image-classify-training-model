import pytest
import numpy as np
from src.data_loader import load_images_and_labels
from src.model_trainer import create_model
from src.predictor import predict_image_class


@pytest.mark.parametrize("input_shape, num_classes", [((150, 150, 3), 3), ((224, 224, 3), 5)])
def test_create_model(input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    assert model is not None, "Model should be created successfully"
    assert model.input_shape[1:] == input_shape, "Model input shape mismatch"
    assert model.output_shape[-1] == num_classes, "Model output class count mismatch"
