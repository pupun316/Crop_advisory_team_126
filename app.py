from flask import Flask, render_template, request
import pickle
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the trained crop recommendation model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise Exception(f"Model loading failed: {e}")

# Crop label to name mapping dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expected input features from the form
        feature_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Extract and convert form data to float
        features = []
        for key in feature_keys:
            value = request.form.get(key)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing input for: {key}")
            features.append(float(value))

        features_array = np.array([features])  # Shape (1, 7)

        # Make prediction
        prediction = model.predict(features_array)[0]

        # Get crop name from dictionary
        crop = crop_dict.get(prediction)
        if crop:
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = " Unable to determine the best crop for these conditions."

        logging.info(f"Prediction made successfully: {crop}")

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        result = f"⚠️ Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
