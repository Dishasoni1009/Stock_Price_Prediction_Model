import joblib

def load_model_and_predict(new_data):
    """
    Load the trained model and make predictions on new data.
    """
    try:
        # Load the trained model
        model = joblib.load("artifacts/model.pkl")
        print("Model loaded successfully!")

        # Make predictions
        predictions = model.predict(new_data)
        print("Predictions:", predictions)

    except Exception as e:
        print("Error during prediction:", e)

if __name__ == "__main__":
    # Example input data (replace this with actual input feature values)
    new_data = [[99690.6,99731.2,99655.2,196.22,4814.0,101.002,10068906.2836,12,13,2024]]  # Replace with actual feature values
    load_model_and_predict(new_data)
