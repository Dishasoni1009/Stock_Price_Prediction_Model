from flask import Flask, render_template, request, jsonify
# from flask import Flask, request, jsonify
# import pandas as pd
# import joblib


from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Load the trained model
MODEL_PATH = os.path.join("artifacts", "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found. Ensure model training has been completed.")
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Stock Price Prediction Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request JSON data
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT')
        interval = int(data.get('interval', 5))
        limit = int(data.get('limit', 1000))

        # Step 1: Fetch and Preprocess Data
        logging.info("Fetching and preprocessing data...")
        ingestion = DataIngestion()
        raw_data = ingestion.fetch_data_from_binance(symbol=symbol, time_interval=interval, limit=limit)
        processed_data = ingestion.preprocess_data(raw_data)
        
        # Step 2: Transform the Data
        logging.info("Scaling the features...")
        transformation = DataTransformation()
        processed_data, _ = transformation.scale_features(processed_data, processed_data)

        # Step 3: Make Predictions
        logging.info("Making predictions...")
        features = processed_data.drop(columns=['close'])
        predictions = model.predict(features)

        # Step 4: Return Response
        return jsonify({
            "predictions": predictions.tolist(),
            "message": "Prediction successful"
        })

    except CustomException as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)})
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
