from flask import Flask, render_template, request, redirect, url_for, Response, session, flash, send_file
import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from pymongo import MongoClient
import bcrypt
from collections import deque
import os
import logging
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure random key

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI if needed
db = client['user_database']  # Replace with your database name
users_collection = db['users']  # Collection to store user credentials

# Load the model
model_path = r'C:\Users\91600\OneDrive\Desktop\Real Time  Threat Detection System\my_flask_app\Suraksha-dhi-Real_Time_Threat_Detection_System-\violence_detection\model\modelnew (1).h5'

try:
    model = load_model(model_path)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    app.logger.info("Model loaded successfully.")
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

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    # Resize and preprocess the frame as required by your model
    frame = cv2.resize(frame, (128, 128))  # Resize to 128x128
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

# Function to display results on the frame
def display_results(frame, label):
    # Draw labels and bounding boxes as needed on the frame
    text = "Violence Detected" if label else "No Violence Detected"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label else (0, 255, 0), 2)  # Swapped colors
    return frame


# Function to capture the laptop camera stream and process for violence detection
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

        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame)

        # Make predictions on the frame
        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0] if model is not None else np.zeros(2)
        Q.append(preds)

        # Perform prediction averaging
        results = np.array(Q).mean(axis=0)
        label = (results > 0.7)[0]  # Adjust threshold as needed

        # Display results on the frame
        output_frame = display_results(frame, label)

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the camera

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
            time.sleep(0.1)  # Add a brief pause before retrying
            continue  # Retry reading the frame
        
        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame)
        
        # Make predictions on the frame
        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0] if model is not None else np.zeros(2)
        Q.append(preds)

        # Perform prediction averaging
        results = np.array(Q).mean(axis=0)
        label = (results > 0.7)[0]  # Adjust threshold as needed

        # Display results on the frame
        output_frame = display_results(frame, label)

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to upload a video file for analysis with error handling
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'username' not in session:
        flash("Please log in to upload videos.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        video_file = request.files.get('video_file')
        if video_file and allowed_file(video_file.filename):
            video_path = os.path.join("uploads", video_file.filename)
            video_file.save(video_path)
            app.logger.debug(f"Video file saved to: {video_path}")
            return redirect(url_for('analyze_video', video_path=video_path))
        else:
            flash("Invalid file type or no file uploaded. Please try again.", "error")
            return redirect(url_for('upload_video'))
    return render_template('upload_video.html')

def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# New route for the laptop camera feed
@app.route('/video_feed_laptop')
def video_feed_laptop():
    return Response(laptop_camera_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_video')
def analyze_video():
    video_path = request.args.get('video_path')
    
    if not video_path or not os.path.isfile(video_path):
        flash("Video file not found. Please upload a valid video.", "error")
        return redirect(url_for('upload_video'))

    app.logger.info(f"Analyzing video at: {video_path}")
    
    # Initialize results and frames list
    frames = []

    # Process the video for analysis
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame)

        # Make predictions on the frame
        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0] if model is not None else np.zeros(2)
        label = preds[0] > 0.7  # Adjust threshold as needed

        # Display results on the frame
        output_frame = display_results(frame, label)

        # Append the processed frame to the frames list
        frames.append(output_frame)

    cap.release()  # Release the video capture

    # Create a video from the frames with results
    output_video_path = os.path.join("uploads", "analyzed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_frame.shape[1], output_frame.shape[0]))

    for f in frames:
        out.write(f)
    out.release()

    # Return the processed video for download or viewing
    return send_file(output_video_path)



if __name__ == '__main__':
    app.run(debug=True)
