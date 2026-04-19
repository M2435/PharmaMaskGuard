# 💊 PharmaMaskGuard 😷

### AI-Driven Mask Compliance Monitoring for Pharmaceutical GMP Environments

---

## 📘 Overview

**PharmaMaskGuard** is an AI-powered mask compliance monitoring system designed for **pharmaceutical manufacturing** and **cleanroom environments**, where contamination control is critical.

By ensuring proper mask usage, the system supports **Good Manufacturing Practice (GMP)** guidelines and reduces the risk of contamination during drug formulation, packaging, and quality control processes.

The project uses a **MobileNetV2-based deep learning model** to classify mask usage into three categories and performs **real-time detection** using webcam or CCTV feeds with OpenCV.

---

## 🎯 Key Features

### ⚡ Real-Time Detection

* Low-latency predictions using OpenCV and Haar Cascade face detection
* Works with webcam or CCTV feeds

### 🔍 3-Class Mask Classification

* ✅ **With Mask** — Fully compliant
* ⚠️ **Improper Mask** — Nose/mouth partially exposed
* ❌ **Without Mask** — Non-compliant

### 🧠 Lightweight & Efficient

* Built using **MobileNetV2**
* Optimized for real-time and edge-device deployment

### 📊 Model Evaluation

* Confusion Matrix
* Accuracy, Precision, Recall, F1-score
* Performance visualization

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras (MobileNetV2)
* **Computer Vision:** OpenCV (Haar Cascade)
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib

---

## 📁 Project Structure

```text
PharmaMaskGuard/
│
├── dataset/                   # (Not included in repo)
│   ├── with_mask/
│   ├── without_mask/
│   └── improper_mask/
│
├── train_mask_detector.py     # Training pipeline (MobileNetV2 + Augmentation)
├── evaluate.py                # Evaluation script (Confusion Matrix + Metrics)
├── realtime.py                # Real-time webcam inference
├── mask_detector_model.h5     # Trained model file
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python train_mask_detector.py
```

### 3️⃣ Evaluate the Model

```bash
python evaluate.py
```

### 4️⃣ Run Real-Time Detection

```bash
python realtime.py
```

---

## 🚫 Dataset Note

The dataset is **not included in this repository** due to size limitations.

Expected folder structure:

```text
dataset/
│── with_mask/
│── without_mask/
└── improper_mask/
```

---

## 📌 Use Cases

* Pharmaceutical production & sterile manufacturing
* Cleanroom compliance monitoring
* GMP regulation enforcement
* Automated PPE verification in industrial environments

---

## 🚀 Future Improvements

* Deployment on edge devices (Raspberry Pi / Jetson Nano)
* Integration with CCTV surveillance systems
* Alert system for non-compliance
* Multi-person tracking and analytics dashboard

---

## 📜 License

This project is open-source. You can use the **MIT License** or any license as per your requirement.

---

## 👨‍💻 Authors

* Your Name
* Team Members (if any)

---
