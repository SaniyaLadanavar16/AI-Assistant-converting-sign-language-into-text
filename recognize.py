import cv2
import numpy as np
from model_training import model

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0
    frame = frame.reshape((1, 64, 64, 1))

    # Make predictions
    predictions = model.predict(frame)
    class_id = np.argmax(predictions)

    # Map class ID to sign language text
    sign_language_text = unique_labels[class_id]

    # Display the recognized text
    cv2.putText(frame, sign_language_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()