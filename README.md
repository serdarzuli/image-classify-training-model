# 🖼️ Image Classification Model with PDF and Image Support

This project is designed to classify images extracted from both PDF documents and standard image files using deep learning models. It leverages TensorFlow, Keras, and PyMuPDF for efficient data handling and accurate predictions.

---

## 🚀 Features

- 📥 **PDF and Image File Support**: Extract and classify images from PDFs and standard image formats.
- 🤖 **Deep Learning Model**: Built using TensorFlow and Keras for robust image classification.
- 🔍 **Multi-Class Prediction**: Supports classification into multiple categories.
- 🗂️ **Modular Code Structure**: Easily maintainable and scalable codebase.
- ✅ **Automated Testing**: Integrated tests for core functionalities.

---

## 📂 Project Structure

```
IMAGE-CLASSIFY-TRAINING-MODEL/
│
├── data/
│   ├── raw/                # Raw data files (PDFs and images)
│   ├── processed/          # Processed and cleaned data
│   ├── models/             # Saved trained models
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Functions for loading and preprocessing data
│   ├── model_trainer.py    # Model building and training logic
│   ├── predictor.py        # Loading model and making predictions
│   ├── utils.py            # Helper functions
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_model_trainer.py
│   ├── test_predictor.py
│
├── notebooks/              # Jupyter Notebooks for analysis
│
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Files ignored by Git
├── train.py                # Script for training the model
├── predict.py              # Script for running predictions
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/image-classification-model.git
   cd image-classification-model
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Usage

### 🎓 Train the Model

Run the following command to train the model with the prepared dataset:

```bash
python train.py
```

### 🔮 Make Predictions

Once the model is trained, use this command to classify a new image:

```bash
python predict.py
```

---

## 🛠️ Configuration

Adjust the following paths in the code as needed:

- **Data Directory:** Update `data/raw/` with your training data.
- **Model Save Path:** Configure the save directory in `train.py`.

---

## ✅ Testing

Run the tests using:

```bash
pytest
```

---

## 📚 Requirements

- TensorFlow
- Keras
- NumPy
- scikit-learn
- PyMuPDF
- Pillow

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- TensorFlow & Keras for deep learning tools
- PyMuPDF for PDF processing
- scikit-learn for machine learning utilities

---

Happy coding! 🚀

