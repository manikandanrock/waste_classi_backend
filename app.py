import os
import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify
import requests
import tensorflow as tf
import numpy as np
import base64
import logging  # Import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ***REPLACE WITH THE ACTUAL, WORKING DOWNLOAD URL***
MODEL_URL = "https://drive.usercontent.google.com/download?id=1FrRjlIOFDK7qWvgOP6ACHuWHfHhYF9sZ&export=download&authuser=0&confirm=t&uuid=1c32b9cf-b137-4bb6-a0be-b2a1f18091d0&at=AIrpjvNHE6QhgliQKSpwFRtKr2r4%3A1739025597271"
MODEL_PATH = "./cnnmodel.h5"
CLASS_NAMES = ['Organic', 'Recycleable']  # Define class names

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model...")  # Use logging
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Check for HTTP errors

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):  # Larger chunk size
                    f.write(chunk)

            file_size = os.path.getsize(MODEL_PATH)
            logging.info(f"Model downloaded. Size: {file_size} bytes")
        except requests.exceptions.RequestException as e:
            logging.error(f"Download error: {e}")
            raise  # Re-raise to stop app initialization
    else:
        file_size = os.path.getsize(MODEL_PATH)
        logging.info(f"Model file already exists. Size: {file_size} bytes")


download_model() # Download model before loading it.

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise  # Stop app initialization if model loading fails


def preprocess_image(file):
    try:
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype('float32')
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise  # Re-raise the exception


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500  # HTTP 500

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400  # HTTP 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400

        logging.info("Processing image...")
        img_array = preprocess_image(file)

        logging.info(f"Image array shape: {img_array.shape}")
        prediction = model.predict(img_array)

        # ***HANDLE MULTI-CLASS PREDICTIONS CORRECTLY***
        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_index]  # Use class names
        predicted_score = prediction[0][predicted_class_index]

        img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")  # Or PNG
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': predicted_class,
            'score': float(predicted_score),
            'image': img_str
        }), 200  # HTTP 200

    except Exception as e:
        logging.exception(f"Error in prediction: {e}")  # Use logging.exception
        return jsonify({'error': str(e)}), 500  # HTTP 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
