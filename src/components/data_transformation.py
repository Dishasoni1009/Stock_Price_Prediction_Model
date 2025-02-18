# import sys
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler

# from src.exception import CustomException
# from src.logger import logging
# import os
# from src.utils import save_object

# @dataclass
# class DataTransformationconfig():
#     preprocessor_obj_file_path=os.path.join('artifacts',"Model.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationconfig()

        
#     def get_data_transformation_object(self):
#         '''
#         this function is responsible for data transformation
#         '''
#         try:
#             numeric_columns = ['open', 'high', 'low', 'volume', 'num_trades','taker_base_vol', 'taker_quote_vol', 'month', 'day', 'year']

#             num_pipeline = Pipeline(
#                 steps=[

#                     ("imputer",SimpleImputer(strategy="median")),
#                     ("scaler",StandardScaler())
#                 ]
#             )

#             logging.info("numerical columns standard scalling completed")
            
#             preprocessor=ColumnTransformer(
#                 [
#                     ("num_pipeline",num_pipeline,numeric_columns)
#                 ]
#             )
#             return preprocessor
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data, and standardized column names.")
#             logging.info("read train and test data complete")
#             logging.info("obtaining preprocesso object")

#             preprocessing_obj=self.get_data_transformation_object()

#             target_column_price='close'
#             numeric_columns = ['open','high','low','close','volume','num_trades','taker_base_vol','taker_quote_vol','month','day','year']
            
#             input_feature_train_df=train_df.drop(columns=[target_column_price],axis=1)
#             target_feature_train_df=train_df[target_column_price]

#             input_feature_test_df=test_df.drop(columns=[target_column_price],axis=1)
#             target_feature_test_df=test_df[target_column_price]

#             logging.info(
#                 f"Applying preprocessing object on training dataframe and testing dataframe."
#             )

#             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df),
#             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

#             train_arr = np.c_[
#                 input_feature_train_arr, np.array(target_feature_train_df)
#             ]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info(f"Saved preprocessing object.")

#             save_object(

#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e,sys)

# /.........................
# import os
# import sys
# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformation_object(self):
#         """
#         This function creates a preprocessor object for numerical transformations.
#         """
#         try:
#             # Define the numerical columns
#             numeric_columns = ['open', 'high', 'low', 'volume', 'num_trades', 
#                                'taker_base_vol', 'taker_quote_vol', 'month', 'day', 'year']

#             # Numerical pipeline
#             num_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="median")),  # Fill missing values
#                     ("scaler", StandardScaler())                   # Scale numerical features
#                 ]
#             )

#             logging.info("Numerical column standard scaling completed")

#             # Combine pipelines into a column transformer
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("num_pipeline", num_pipeline, numeric_columns)
#                 ]
#             )

#             return preprocessor
#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_data_transformation(self, train_df, test_df):
#         """
#         Apply transformations to train and test datasets.
#         """
#         try:
#             logging.info("Obtaining preprocessing object")

#             preprocessing_obj = self.get_data_transformation_object()

#             target_column = 'close'
#             input_features = train_df.drop(columns=[target_column], axis=1)
#             target_features = train_df[target_column]

#             input_features_test = test_df.drop(columns=[target_column], axis=1)
#             target_features_test = test_df[target_column]

#             logging.info("Applying preprocessing object to training and testing datasets")

#             input_features_train_arr = preprocessing_obj.fit_transform(input_features)
#             input_features_test_arr = preprocessing_obj.transform(input_features_test)

#             train_array = np.c_[
#                 input_features_train_arr, np.array(target_features)
#             ]
#             test_array = np.c_[
#                 input_features_test_arr, np.array(target_features_test)
#             ]

#             logging.info(f"Saving preprocessing object at {self.data_transformation_config.preprocessor_obj_file_path}")
#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )

#             return (
#                 train_array,
#                 test_array,
#                 self.data_transformation_config.preprocessor_obj_file_path
#             )
#         except Exception as e:
#             raise CustomException(e, sys)

# /..............................

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    """
    Configuration for Data Transformation.
    """
    transformed_train_data_file_path: str = os.path.join("artifacts", "transformed_train.csv")
    transformed_test_data_file_path: str = os.path.join("artifacts", "transformed_test.csv")

class DataTransformation:
    """
    Class for handling data transformation, including feature scaling and preparing datasets for model training.
    """

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Scale numerical features using StandardScaler.

        Parameters:
        - train_df: Training dataset (DataFrame)
        - test_df: Testing dataset (DataFrame)

        Returns:
        - Scaled training and testing DataFrames
        """
        try:
            logging.info("Starting feature scaling")

            # Separate features and target
            X_train = train_df.drop(columns=['close'])
            y_train = train_df['close']
            X_test = test_df.drop(columns=['close'])
            y_test = test_df['close']

            # Initialize scaler
            scaler = StandardScaler()

            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Combine scaled features with target
            train_transformed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            train_transformed['close'] = y_train.reset_index(drop=True)

            test_transformed = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            test_transformed['close'] = y_test.reset_index(drop=True)

            logging.info("Feature scaling completed")
            return train_transformed, test_transformed

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Perform data transformation and save transformed datasets to CSV.

        Parameters:
        - train_df: Training dataset (DataFrame)
        - test_df: Testing dataset (DataFrame)

        Returns:
        - Transformed training and testing DataFrames
        """
        try:
            logging.info("Starting data transformation process")

            # Scale features
            train_transformed, test_transformed = self.scale_features(train_df, test_df)

            # Save transformed data to CSV
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_data_file_path), exist_ok=True)
            train_transformed.to_csv(self.transformation_config.transformed_train_data_file_path, index=False)
            test_transformed.to_csv(self.transformation_config.transformed_test_data_file_path, index=False)

            logging.info(f"Transformed training data saved to {self.transformation_config.transformed_train_data_file_path}")
            logging.info(f"Transformed testing data saved to {self.transformation_config.transformed_test_data_file_path}")

            return train_transformed, test_transformed

        except Exception as e:
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    # Example placeholders for input datasets (replace with actual paths or DataFrames in practice)
    ingestion_artifacts_path = "artifacts"
    train_path = os.path.join(ingestion_artifacts_path, "train.csv")
    test_path = os.path.join(ingestion_artifacts_path, "test.csv")

    try:
        # Read input datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Initialize and run DataTransformation
        data_transformation = DataTransformation()
        transformed_train, transformed_test = data_transformation.initiate_data_transformation(train_df, test_df)

    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        raise CustomException(e, sys)

