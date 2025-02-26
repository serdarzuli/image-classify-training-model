from src.predictor import predict_image_class

model_path = 'data/models/final_model.keras'
image_path = 'data/raw/test/towing_cost.jpg' 

predicted_class_index, predicted_class_name, confidence = predict_image_class(model_path, image_path)
print(f"The image is predicted to be in class: {predicted_class_index}, the name is {predicted_class_name} with confidence {confidence:.2%}")
