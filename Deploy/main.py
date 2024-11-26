import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow import keras
import face_recognition
import cv2
import numpy as np

# Flask app configuration
app = Flask(__name__)

# Set the secret key for session management
app.secret_key = os.urandom(24)  # Generate a random secret key for session management

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the upload folder in Flask app config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure the uploads folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(file_path)

        flash('Video successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Invalid file type. Please upload a valid video file.')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/predict/<filename>')
def sequence_prediction(filename):
    try:
        # Load the pre-trained model with an updated path
        sequence_model = load_model(os.path.join(os.getcwd(), 'models', 'inceptionNet_model.h5'))
        class_vocab = ['FAKE', 'REAL']

        # Load video frames
        frames = load_video(f'static/uploads/{filename}')
        if frames.size == 0:
            flash("Error: Unable to process video.")
            return render_template('upload.html', filename=filename, prediction="ERROR")

        frame_features, frame_mask = prepare_single_video(frames)

        # Predict using the model
        probabilities = sequence_model.predict([frame_features, frame_mask])[0]
        pred = probabilities.argmax()

        return render_template('upload.html', filename=filename, prediction=class_vocab[pred])
    except Exception as e:
        flash(f"Error: {str(e)}")
        return render_template('upload.html', filename=filename, prediction="ERROR")

# Helper Functions
def allowed_file(filename):
    """Check if the uploaded file is a valid video file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    """Load the video and extract frames."""
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / SEQ_LENGTH), 1)
    frames = []

    try:
        for frame_cntr in range(SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames_window)
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_face_center(frame)
            if frame is None:
                continue
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    return np.array(frames)

def prepare_single_video(frames):
    """Preprocess the video frames.""" 
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def build_feature_extractor():
    """Build the feature extractor model using InceptionV3.""" 
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def crop_face_center(frame):
    """Crop the face from the frame using face recognition."""
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return None  # Return None if no face is detected

    top, right, bottom, left = face_locations[0]
    return frame[top:bottom, left:right]

if __name__ == "__main__":
    app.run(debug=True)