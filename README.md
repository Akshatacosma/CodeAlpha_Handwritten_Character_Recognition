# CodeAlpha_Handwritten_Character_Recognition
Handwritten digit recognition using CNN and MNIST dataset. Built as part of CodeAlpha Machine Learning Internship.
# ✍️ Handwritten Character Recognition using CNN

## 📌 Objective

The objective of this project is to identify handwritten digits using deep learning techniques. This system can recognize digits from images and classify them accurately.

---

## 🧠 Project Overview

Handwritten Character Recognition is a computer vision task where a model is trained to recognize digits or characters from handwritten images. In this project, we use the MNIST dataset and a Convolutional Neural Network (CNN) to achieve high accuracy in digit classification.

---

## ⚙️ Technologies Used

* Python 🐍
* TensorFlow / Keras 🤖
* NumPy
* Matplotlib

---

## 📊 Dataset

* **MNIST Dataset**

  * 70,000 grayscale images
  * Image size: 28 × 28 pixels
  * Classes: Digits (0–9)

---

## 🧩 Approach

### 🔹 Data Preprocessing

* Normalized pixel values (0–255 → 0–1)
* Reshaped data for CNN input

### 🔹 Model Architecture

* Convolutional Layers (feature extraction)
* Max Pooling Layers (dimensionality reduction)
* Fully Connected (Dense) Layers
* Softmax Output Layer (classification)

---

## 🤖 Model Used

* Convolutional Neural Network (CNN)

---

## 📈 Performance

* Achieved accuracy of approximately **98%–99%** on test data

---

## 🚀 How It Works

1. Load dataset (MNIST)
2. Preprocess images
3. Train CNN model
4. Predict handwritten digits
5. Display results

---

## ▶️ How to Run

```bash
pip install tensorflow numpy matplotlib
python handwritten_recognition.py
```

---

## 📷 Output

* Displays input image
* Predicts digit (0–9)

---

## 💡 Applications

* Digit recognition (bank cheques, postal codes)
* Form digitization
* Automated data entry systems

---

## 👩‍💻 Author

Akshata

---

## ⭐ Acknowledgment

This project is part of the CodeAlpha Machine Learning Internship.
