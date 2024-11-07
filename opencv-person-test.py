import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define class labels
class_labels = ['Anthony', 'Obama', 'Catherina', 'Bush']

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

# Load the Haar Cascade for face detection
cascade_path = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Resize windows to a good size
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.array(face_roi) / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)

        predictions = model.predict(face_roi)
        best_class_idx = np.argmax(predictions)
        best_class = class_labels[best_class_idx]
        confidence = np.max(predictions)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = '{}: {:.2f}%'.format(best_class, confidence * 100)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()