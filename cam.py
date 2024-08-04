import cv2

# Open the default camera (index 0) with AVFoundation backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # If a frame was captured, display it
    if ret:
        cv2.imshow('Camera', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()