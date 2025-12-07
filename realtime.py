import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('mask_detector_model.h5')

# Class labels (match your dataset folder names)
labels = ["ImproperMask", "WithMask", "WithoutMask"]

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict mask status
        preds = model.predict(face_input)
        label_index = np.argmax(preds)
        label = labels[label_index]
        confidence = preds[0][label_index]

        # Draw bounding box + label
        color = (0, 255, 0) if label == "WithMask" else ((0, 255, 255) if label == "ImproperMask" else (0, 0, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
