import os
import numpy as np
from PIL import Image
import fitz  # PyMuPDF


def preprocess_image(image, target_size=(150, 150)):
    """  Make 150x150 size and normalization. """
    img = image.convert("RGB")  # if file is black&white, convert it to  RGB
    img = img.resize(target_size)  # 150x150 verbose
    img = np.array(img) / 255.0  # Normalization
    return img


def load_images_and_labels(data_dir, target_size=(150, 150)):
    images, labels = [], []
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if not os.path.isfile(file_path):
                continue
            
            try:
                # if file is a pdf convert it to as img
                if file_path.lower().endswith(".pdf"):
                    doc = fitz.open(file_path)  # Open pdf file
                    for page_num in range(len(doc)):  # Process all pages
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()  # page to img
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img = preprocess_image(img, target_size)  # start with preprocess
                        images.append(img)
                        labels.append(category)  # save the category
                    doc.close()
                
                # start according if the file is an image 
                elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = Image.open(file_path)
                    img = preprocess_image(img, target_size)  
                    images.append(img)
                    labels.append(category)

            except Exception as e:
                print(f"🚨 Hata: {file_path} işlenemedi -> {e}")

    return np.array(images), np.array(labels)


def process_pdf(file_path, images, labels, category, target_size):
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.resize(target_size)
        images.append(np.array(img) / 255.0)
        labels.append(category)
    doc.close()
    return images, labels


def process_image(file_path, images, labels, category, target_size):
    image = Image.open(file_path).resize(target_size)
    images.append(np.array(image) / 255.0)
    labels.append(category)
    return images, labels
