# Indian Rupee Counterfeit Detection System

## 📖 Overview

The **Indian Rupee Counterfeit Detection System** is a Machine Learning and Computer Vision project developed to classify Indian currency notes as **Real** or **Fake**.

Currently, the project supports **₹100** and **₹500** currency notes. Images are preprocessed, organized into training and testing datasets, and prepared for a Convolutional Neural Network (CNN) model built using TensorFlow and Keras.

This repository currently includes all work completed up to **Day 5**, including dataset preparation, preprocessing, data loading, and CNN model development.

---

## ✨ Features Completed

- Dataset organization for ₹100 and ₹500 notes
- Image preprocessing using OpenCV
- Image resizing to **224 × 224**
- Dataset splitting (80% Training / 20% Testing)
- Image normalization (pixel values scaled to 0–1)
- Custom CNN architecture using TensorFlow/Keras
- Clean project structure for future development

---

## 🛠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## 📁 Project Structure

```text
Indian-Rupee-Counterfeit-Detection/
│
├── dataset/
│   ├── processed/
│   │   ├── train/
│   │   └── test/
│
├── docs/
├── models/
├── notebooks/
├── results/
├── screenshots/
├── src/
│   ├── preprocess.py
│   ├── data_pipeline.py
│   ├── dataset_loader.py
│   └── model.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The dataset contains images of Indian currency notes categorized as:

- Real ₹100
- Fake ₹100
- Real ₹500
- Fake ₹500

The dataset is preprocessed by resizing all images to **224 × 224** pixels before being split into training and testing sets.

---

## 🚀 Current Project Status

**Completed (Day 1–Day 5)**

- Project setup
- Dataset collection and organization
- Image preprocessing
- Dataset splitting
- Dataset loading pipeline
- CNN model architecture

**Upcoming Work**

- Model training
- Performance evaluation
- Confusion matrix
- Classification report
- Prediction module
- Webcam integration
- Streamlit web application

---

