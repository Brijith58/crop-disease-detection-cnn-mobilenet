# 🌿 Crop Disease Detection using CNN and MobileNetV2

## 📌 Overview

This project performs plant disease classification using deep learning techniques. A comparative study is conducted between a **Convolutional Neural Network (CNN)** and **MobileNetV2 (Transfer Learning)** to evaluate performance on crop disease detection.

---

## 🚀 Features

* CNN model built from scratch
* MobileNetV2 pretrained transfer learning model
* Accuracy and loss visualization
* Confusion matrix and classification report
* Flask web application for real-time prediction

---

## 🧠 Models Used

* CNN (Custom Architecture)
* MobileNetV2 (Transfer Learning)

---

## 📊 Results

| Model       | Accuracy |
| ----------- | -------- |
| CNN         | 87.6%    |
| MobileNetV2 | 96.0%    |

---

## 📸 Sample Outputs

*outputs/
* Accuracy Graph
* Loss Graph
* Confusion Matrix

---

## ⚙️ Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train Models

```
python train_cnn.py
python train.py
```

### Run Web App

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📁 Dataset

The dataset is not included due to size limitations.

Download from:
https://www.kaggle.com/datasets/emmarex/plantdisease

Place it inside:

```
dataset/
```

---

## 📂 Project Structure

```
crop-disease-detection/
│
├── train.py
├── train_cnn.py
├── app.py
├── models/
├── outputs/
├── templates/
├── static/
├── requirements.txt
├── README.md
```

---

## 📄 Research Work

This project is part of a comparative study between CNN and MobileNetV2 for crop disease classification.

---

## 👨‍💻 Authors

* Brijith Manikandan

---

## ⭐ Acknowledgment

Dataset sourced from PlantVillage (Kaggle)

---
