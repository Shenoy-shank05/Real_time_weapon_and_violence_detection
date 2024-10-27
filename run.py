from flask import Flask, render_template, request, redirect, url_for, Response, session, flash, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
import bcrypt
from collections import deque
import os
import logging
import time

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use environment variable for secret key

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI if needed
db = client['user_database']  # Replace with your database name
users_collection = db['users']  # Collection to store user credentials

# Load the YOLO model for weapon detection
model_path = os.getenv('YOLO_MODEL_PATH', r'C:\Users\91600\OneDrive\Desktop\Real Time  Threat Detection System\my_flask_app\Suraksha-dhi-Real_Time_Threat_Detection_System-\weapon_detection\best.pt')  # Use environment variable for YOLO model path

try:
    model = YOLO(model_path)  # Load the YOLO model
    app.logger.info("Weapon detection model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

Q = deque(maxlen=128)

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check credentials from MongoDB
        user = users_collection.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['username'] = username  # Store user session
            return redirect(url_for('choose_feed'))
        else:
            flash("Invalid credentials. Please try again.", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

# Route for the sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists in MongoDB
        if users_collection.find_one({"username": username}):
            flash("Username already exists. Please choose a different one.", "error")
            return redirect(url_for('signup'))
        else:
            # Hash the password before saving it to the database
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            users_collection.insert_one({"username": username, "password": hashed_password.decode('utf-8')})
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
    return render_template('signup.html')

# Ensure the user is logged in before accessing feed options
@app.route('/choose_feed')
def choose_feed():
    if 'username' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
    return render_template('choose_feed.html')

# Route for the feed configuration page (for RTSP URL)
@app.route('/feed', methods=['GET', 'POST'])
def feed():
    if 'username' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        rtsp_url = request.form['rtsp_url']
        session['rtsp_url'] = rtsp_url  # Store RTSP URL in session
        return redirect(url_for('video_feed_cctv'))  # Redirect to CCTV feed
    return render_template('feed.html')

# Function to display bounding boxes and results on the frame
def display_results(frame, results):
    # Loop over each detection
    for detection in results:
        # Extract bounding box coordinates and confidence
        x1, y1, x2, y2, conf, cls = detection[:6]
        
        # Draw the bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # Add label text for the class (Weapon) and confidence score
        label = f"Weapon: {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return frame

# Function to capture the laptop camera stream and process for weapon detection
def laptop_camera_stream():
    app.logger.info("Opening laptop camera...")
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        app.logger.error("Unable to open laptop camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            app.logger.error("Unable to read frame from laptop camera. Retrying...")
            time.sleep(0.1)  # Add a brief pause before retrying
            continue  # Retry reading the frame

        # Resize the frame for consistent processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Make predictions on the frame
        if model:
            try:
                results = model(frame_resized)[0].boxes.data.cpu().numpy()  # Run the YOLO model on the frame and get detections
            except Exception as e:
                app.logger.error(f"Prediction error: {e}")
                results = []
        else:
            results = []  # Default to no detections if the model is not loaded

        output_frame = display_results(frame_resized, results)

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Function to capture CCTV stream using RTSP
def cctv_camera_stream():
    rtsp_url = session.get('rtsp_url', None)
    if rtsp_url is None:
        app.logger.error("No RTSP URL provided.")
        return b''  # No RTSP URL provided
    
    cap = cv2.VideoCapture(rtsp_url)  # Connect to CCTV camera using RTSP
    if not cap.isOpened():
        app.logger.error(f"Unable to open RTSP stream at {rtsp_url}")
        return b''  # Unable to open RTSP stream
    
    while True:
        ret, frame = cap.read()
        if not ret:
            app.logger.error("No frame received from RTSP stream. Retrying...")
            time.sleep(0.1)
            continue

        frame_resized = cv2.resize(frame, (640, 480))

        if model:
            try:
                results = model(frame_resized)[0].boxes.data.cpu().numpy()  # Get bounding boxes for detections
            except Exception as e:
                app.logger.error(f"Prediction error: {e}")
                results = []
        else:
            results = []

        output_frame = display_results(frame_resized, results)

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to upload a video file for weapon detection
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'username' not in session:
        flash("Please log in to upload videos.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        video_file = request.files.get('video_file')
        if video_file and allowed_file(video_file.filename):
            file_path = os.path.join('uploads', video_file.filename)
            video_file.save(file_path)
            session['uploaded_video'] = file_path
            return redirect(url_for('video_feed_uploaded'))
    return render_template('upload_video.html')

# Function to capture and process uploaded video for weapon detection
def uploaded_video_stream(video_path):
    if video_path is None or not os.path.exists(video_path):
        app.logger.error("No valid uploaded video found.")
        return b''  # No valid uploaded video found
    
    cap = cv2.VideoCapture(video_path)  # Open the uploaded video
    if not cap.isOpened():
        app.logger.error(f"Unable to open uploaded video at {video_path}")
        return b''  # Unable to open the video file
    
    while True:
        ret, frame = cap.read()
        if not ret:
            app.logger.info("End of video reached or no frame received.")
            break  # End of the video file

        if model:
            try:
                results = model(frame)[0].boxes.data.cpu().numpy()  # Get bounding boxes for detections
            except Exception as e:
                app.logger.error(f"Prediction error: {e}")
                results = []
        else:
            results = []

        output_frame = display_results(frame, results)  # Display results on the frame

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



# Route for laptop camera feed
@app.route('/video_feed_laptop')
def video_feed_laptop():
    return Response(laptop_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for RTSP camera feed
@app.route('/video_feed_cctv')
def video_feed_cctv():
    return Response(cctv_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for uploaded video feed
@app.route('/video_feed_uploaded')
def video_feed_uploaded():
    video_path = session.get('uploaded_video', None)
    if video_path:
        return Response(uploaded_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        flash("No uploaded video found.", "error")
        return redirect(url_for('upload_video'))  # Redirect if no video is found

@app.route('/logout')
def logout():
    session.clear()  # Clear user session
    return redirect(url_for('login'))

# Function to check allowed file types for video uploads
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['mp4', 'avi', 'mkv', 'mov']

if __name__ == '__main__':
    app.run(debug=True)
