from flask import Flask, render_template, request, redirect, url_for, Response, session
import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from pymongo import MongoClient
import bcrypt
from collections import deque
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI if needed
db = client['user_database']  # Replace with your database name
users_collection = db['users']  # Collection to store user credentials

# Load the model
model_path = 'C:/Users/91600/OneDrive/Desktop/Real Time  Threat Detection System/my_flask_app/Suraksha-dhi-Real_Time_Threat_Detection_System-/violence_detection/model/modelnew.h5'
model = load_model(model_path)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
Q = deque(maxlen=128)


# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check credentials from MongoDB
        user = users_collection.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return redirect(url_for('choose_feed'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('login.html')

# Route for the sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists in MongoDB
        if users_collection.find_one({"username": username}):
            return "Username already exists. Please choose a different one."
        else:
            # Hash the password before saving it to the database
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            # Insert the new user into the MongoDB collection
            users_collection.insert_one({"username": username, "password": hashed_password.decode('utf-8')})
            return redirect(url_for('login'))  # Redirect to login after signup
    return render_template('signup.html')

# Route for the camera feed choice page
@app.route('/choose_feed')
def choose_feed():
    return render_template('choose_feed.html')

# Route for the live feed configuration page
@app.route('/feed', methods=['GET', 'POST'])
def feed():
    if request.method == 'POST':
        rtsp_url = request.form['rtsp_url']
        session['rtsp_url'] = rtsp_url  # Store RTSP URL in session
        return redirect(url_for('video_feed_cctv'))  # Redirect to CCTV feed
    return render_template('feed.html')

# Function to capture the laptop camera stream and process for violence detection
def laptop_camera_stream():
    cap = cv2.VideoCapture(0)  # Open the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame)
        
        # Make predictions on the frame
        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0]
        Q.append(preds)

        # Perform prediction averaging
        results = np.array(Q).mean(axis=0)
        label = (results > 0.52)[0]  # Adjust threshold as needed

        # Display results on the frame
        output_frame = display_results(frame, label)

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    return frame / 255.0  # Normalize frame

def display_results(frame, label):
    text_color = (0, 255, 0) if not label else (0, 0, 255)  # Green for non-violence, red for violence
    text = "Violence: {}".format(label)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (35, 50), FONT, 1.25, text_color, 3)
    return frame

# Function to capture CCTV stream using RTSP
def cctv_camera_stream():
    rtsp_url = session.get('rtsp_url', None)
    if rtsp_url is None:
        return b''  # No RTSP URL provided
    cap = cv2.VideoCapture(rtsp_url)  # Connect to CCTV camera using RTSP
    if not cap.isOpened():
        print(f"Error: Unable to open RTSP stream at {rtsp_url}")
        return b''  # Unable to open RTSP stream
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No frame received from RTSP stream")
            break
        
        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame)
        
        # Make predictions on the frame
        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0]
        Q.append(preds)

        # Perform prediction averaging
        results = np.array(Q).mean(axis=0)
        label = (results > 0.52)[0]  # Adjust threshold as needed

        # Display results on the frame
        output_frame = display_results(frame, label)

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to serve the video stream from the laptop camera
@app.route('/video_feed_laptop')
def video_feed_laptop():
    return Response(laptop_camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the video stream from the CCTV camera
@app.route('/video_feed_cctv')
def video_feed_cctv():
    return Response(cctv_camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
