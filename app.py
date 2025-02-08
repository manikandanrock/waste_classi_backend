import os
import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify
import requests
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import base64
import zipfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_URL = "https://drive.google.com/uc?export=download&id=1FrRjlIOFDK7qWvgOP6ACHuWHfHhYF9sZ"
MODEL_PATH = "./model/cnnmodel.h5"

# Function to download the model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        os.makedirs("./model", exist_ok=True)
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Model downloaded successfully.")

# Download the model before loading
download_model()

# Load your trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
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

        # Validate file extension
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload an image file (jpg, jpeg, png).'})
        
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
    app.run(debug=True, host="0.0.0.0", port=5000)
