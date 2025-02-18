# import os
# import sys
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from dataclasses import dataclass

# from src.exception import CustomException
# from src.logger import logging

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

# class ModelTrainer:
#     def __init__(self):
#         self.config = ModelTrainerConfig()

#     def train_model(self, X_train, y_train):
#         try:
#             logging.info("Training RandomForestRegressor model...")
#             model = RandomForestRegressor(random_state=42)
#             model.fit(X_train, y_train)
#             logging.info("Model training completed.")
#             return model
#         except Exception as e:
#             raise CustomException(e, sys)

#     def evaluate_model(self, model, X_test, y_test):
#         try:
#             logging.info("Evaluating model performance...")
#             predictions = model.predict(X_test)
#             mse = mean_squared_error(y_test, predictions)
#             logging.info(f"Mean Squared Error: {mse}")
#             return mse
#         except Exception as e:
#             raise CustomException(e, sys)

#     def save_model(self, model):
#         try:
#             logging.info(f"Saving model to {self.config.trained_model_file_path}...")
#             os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
#             with open(self.config.trained_model_file_path, 'wb') as f:
#                 pickle.dump(model, f)
#             logging.info("Model saved successfully.")
#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_model_training(self, train_array, test_array):
#         try:
#             logging.info("Starting model training process...")

#             # Extract features and targets from training and testing arrays
#             X_train, y_train = train_array[:, :-1], train_array[:, -1]
#             X_test, y_test = test_array[:, :-1], test_array[:, -1]

#             # Train the model
#             model = self.train_model(X_train, y_train)

#             # Evaluate the model
#             mse = self.evaluate_model(model, X_test, y_test)

#             # Save the model
#             self.save_model(model)

#             logging.info("Model training process completed successfully.")
#             return mse
#         except Exception as e:
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         from src.components.data_ingestion import DataIngestion
#         from src.components.data_transformation import DataTransformation

#         # Step 1: Data Ingestion
#         data_ingestion = DataIngestion()
#         train_df, test_df = data_ingestion.initiate_data_ingestion()

#         # Step 2: Data Transformation
#         data_transformation = DataTransformation()
#         train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_df, test_df)

#         # Step 3: Model Training
#         model_trainer = ModelTrainer()
#         mse = model_trainer.initiate_model_training(train_array, test_array)

#         # # Calculate training and testing accuracy
#         # training_accuracy = RF.score(X_train, y_train)
#         # testing_accuracy = RF.score(X_test, y_test)

#         # print('Training Accuracy:', training_accuracy)
#         # print('Testing Accuracy:', testing_accuracy)

#         print(f"Model training completed with Mean Squared Error: {mse}")
#     except Exception as e:
#         logging.error(f"Error: {e}")
#         print(f"Error: {e}")

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import joblib

@dataclass
class ModelTrainerConfig:
    """
    Configuration for Model Trainer.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def train_model(self, X_train, y_train):
        """
        Train the Random Forest Regressor model.
        """
        try:
            logging.info("Training the Random Forest Regressor model")
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            logging.info("Model training completed")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model and calculate metrics.
        """
        try:
            logging.info("Evaluating the trained model")
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model Evaluation Completed - MSE: {mse}, R2 Score: {r2}")

            return mse, r2
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, model):
        """
        Save the trained model to a file.
        """
        try:
            os.makedirs(os.path.dirname(self.trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(model, self.trainer_config.trained_model_file_path)
            logging.info(f"Model saved to {self.trainer_config.trained_model_file_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_data_path, test_data_path):
        """
        Perform end-to-end model training and evaluation.
        """
        try:
            logging.info("Starting model training process")

            # Load train and test datasets
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            # Separate features and target
            X_train = train_data.drop(columns=["close"])
            y_train = train_data["close"]

            X_test = test_data.drop(columns=["close"])
            y_test = test_data["close"]

            # Train the model
            model = self.train_model(X_train, y_train)

            # Evaluate the model
            mse, r2 = self.evaluate_model(model, X_test, y_test)

            # Save the trained model
            self.save_model(model)

            return mse, r2

        except Exception as e:
            raise CustomException(e, sys)

# .......................................

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    train_file_path = os.path.join("artifacts", "train.csv")
    test_file_path = os.path.join("artifacts", "test.csv")

    mse, r2 = model_trainer.initiate_model_training(train_file_path, test_file_path)

    print(f"Model Training Completed. MSE: {mse}, R2 Score: {r2}")

