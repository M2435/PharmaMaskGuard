# PharmaMaskGuard
AI-Driven Mask Compliance Monitoring for Pharmaceutical GMP Standards

## Project Overview
PharmaMaskGuard is a computer vision system designed to automate mask compliance monitoring in pharmaceutical manufacturing environments. Strictly adhering to Good Manufacturing Practices (GMP) is critical to prevent contamination in sterile drug production.

This system uses a MobileNetV2 deep learning model to detect face mask usage in real-time via CCTV or webcam feeds, classifying individuals into three categories to ensure maximum safety.

## Key Features
- Real-Time Detection: Low-latency inference using OpenCV and Haar Cascades.
- Three-Class Classification:
    - With Mask: Fully compliant.
    - Improper Mask: Nose exposed or incorrect wear (Critical for GMP).
    - Without Mask: Non-compliant violation.
- Lightweight Architecture: Built on MobileNetV2 for easy deployment on edge devices.
- Comprehensive Evaluation: Includes scripts for confusion matrix and classification reports.

## Tech Stack
- Language: Python 3.x
- Deep Learning: TensorFlow / Keras (MobileNetV2)
- Computer Vision: OpenCV (Haar Cascades)
- Data Processing: NumPy, Pandas, Scikit-learn
- Visualization: Matplotlib

## Project Structure
PharmaMaskGuard/
│
├── dataset/                   # (Not included in repo)
│   ├── with_mask/
│   ├── without_mask/
│   └── improper_mask/
│
├── train_mask_detector.py     # Training pipeline (MobileNetV2 + Augmentation)
├── evaluate.py                # Evaluation script (Confusion Matrix + Metrics)
├── realtime.py                # Real-time webcam inference script
├── mask_detector_model.h5     # Trained model file
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

## Installation and Setup

1. Clone the repository:
   git clone https://github.com/M2435/PharmaMaskGuard.git
   cd PharmaMaskGuard

2. Create a Virtual Environment (Recommended):
   python -m venv venv
   .\venv\Scripts\activate

3. Install Dependencies:
   pip install -r requirements.txt

## Usage Guide

### 1. Training the Model
To retrain the model on a custom dataset, organize your images into the dataset folder and run:
python train_mask_detector.py

### 2. Evaluating Performance
To generate a classification report and confusion matrix:
python evaluate.py

### 3. Real-Time Detection
To start the webcam and execute live inference:
python realtime.py
(Press 'q' to quit the video window)

## Methodology
1. Face Detection: Uses OpenCV Haar Cascade Classifiers to locate faces in the video frame.
2. Preprocessing: Detected faces are resized to 224x224 and preprocessed for MobileNetV2.
3. Classification: The model predicts the probability of the three classes.
4. Alerting: Bounding boxes are color-coded (Green=Safe, Red=Danger, Orange=Warning).

## License
This project is for educational and industrial safety demonstration purposes.
