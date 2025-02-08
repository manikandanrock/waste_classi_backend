import os
import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import base64
import zipfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Authenticate
api = KaggleApi()
api.authenticate()

# Download model from Kaggle
api.dataset_download_file('manikandanvistas/cnnmodel', file_name='cnnmodel.h5', path='./model')

# Load your trained model
try:
    model = tf.keras.models.load_model('./model/cnnmodel.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define image preprocessing function
def preprocess_image(file):
    try:
        # Convert FileStorage object to a BytesIO object
        img_bytes = io.BytesIO(file.read())  # Read the file and convert it to BytesIO
        img = Image.open(img_bytes)  # Open the image using PIL
        img = img.convert('RGB')  # Ensure image is in RGB mode
        img = img.resize((224, 224))  # Resize image to the input size of the model
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32')  # Ensure correct type for processing
        img_array /= 255.0   # Normalize the image to range [0, 1]
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise e

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found, please check the model path.'})

    try:
        # Check if the file is provided in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'})

        file = request.files['file']

        # If no file is selected by the user
        if file.filename == '':
            return jsonify({'error': 'No selected file.'})

        # Process the image
        print("Processing image...")
        img_array = preprocess_image(file)  # Use the updated preprocessing function

        # Predict the class
        print(f"Image array shape: {img_array.shape}")
        prediction = model.predict(img_array)

        # Assuming binary classification: prediction[0][0] > 0.5 means 'Organic', else 'Recyclable'
        predicted_class = 'Organic' if prediction[0][0] > 0.5 else 'Recyclable'
        predicted_score = prediction[0][0] if predicted_class == 'Organic' else 1 - prediction[0][0]

        # Convert image back to uint8 for saving and base64 conversion
        img = Image.fromarray((img_array[0] * 255).astype(np.uint8))  # Convert back to uint8 for displaying
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': predicted_class,
            'score': float(predicted_score),  # Convert float32 to Python float
            'image': img_str
        })

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
