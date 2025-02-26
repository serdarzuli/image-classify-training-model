from src.data_loader import load_images_and_labels
from src.model_trainer import create_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

#load data
train_data_dir = 'data/raw/train'
X_train, y_train = load_images_and_labels(train_data_dir)

# make encode tags
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

model = create_model(input_shape=(150, 150, 3), num_classes=len(set(y_train)))
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

model.save('data/models/final_model.keras')
