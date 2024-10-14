import os
import cv2
import numpy as np
from keras.models import load_model

# Define the path to your model
model_path = "C:\\Users\\91600\\OneDrive\\Desktop\\Real Time Threat Detection System\\my_flask_app\\Suraksha-dhi-Real_Time_Threat_Detection_System-\\violence_detection\\model\\modelnew.h5"

# Load the model
model = load_model(model_path)

# Video input path
video_path = input("Enter the video path or press Enter to use default: ")
if not video_path:
    video_path = "C:\\Users\\91600\\OneDrive\\Desktop\\Real Time Threat Detection System\\my_flask_app\\Suraksha-dhi-Real_Time_Threat_Detection_System-\\violence_detection\\V_19.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the model input size
    processed_frame = cv2.resize(frame, (128, 128))
    processed_frame = processed_frame / 255.0  # Normalize the image
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

    # Get prediction from the model
    prediction = model.predict(processed_frame)

    # Interpret the prediction
    if prediction[0][0] > 0.5:
        label = "Violence Detected"
    else:
        label = "No Violence Detected"

    # Display the result on the video frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Threat Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
