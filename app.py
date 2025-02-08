import os
import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify
import requests
import tensorflow as tf
import numpy as np
import base64
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ***REPLACE WITH THE ACTUAL, WORKING DOWNLOAD URL***
MODEL_URL = "https://drive.usercontent.google.com/download?id=1FMvhvLE2ikEmeIgNFMM8jlnpI_iRSTIn"  # URL for your .tflite model
MODEL_PATH = "./model.tflite"  # Path for the .tflite model
CLASS_NAMES = ['Organic', 'Recycleable']  # Define class names

import gdown

MODEL_URL = "https://drive.google.com/file/d/1FMvhvLE2ikEmeIgNFMM8jlnpI_iRSTIn/view?usp=drive_link"  # Your Google Drive view link
MODEL_PATH = "./model.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model file '{MODEL_PATH}' not found. Downloading...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)  # Download the file
            file_size = os.path.getsize(MODEL_PATH)
            logging.info(f"Model downloaded. Size: {file_size} bytes")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            raise
    else:
        logging.info(f"Model file '{MODEL_PATH}' already exists. Skipping download.")

# ... rest of your code ...

download_model()

try:
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")

except Exception as e:
    logging.error(f"Error loading TFLite model: {e}")
    raise

def preprocess_image(file):
    try:
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB').resize((224, 224))  # Resize to match your model's input
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype(np.float32)  # Important: float32 for TFLite
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:  # Check if interpreter is loaded
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400

        logging.info("Processing image...")
        img_array = preprocess_image(file)

        logging.info(f"Image array shape: {img_array.shape}")

        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_index]
        predicted_score = prediction[0][predicted_class_index]  # Access the score

        img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")  # Or PNG
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': predicted_class,
            'score': float(predicted_score),
            'image': img_str
        }), 200

    except Exception as e:
        logging.exception(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
