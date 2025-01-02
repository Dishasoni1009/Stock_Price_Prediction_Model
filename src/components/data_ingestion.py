# import os 
# import sys 
# from src.exception import CustomException
# from src.logger import logging
# import pandas as pd


# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass
# from src.components.data_transformation import DataTransformation 
# from src.components.data_transformation import DataTransformationconfig

# @dataclass
# class DataIngestionsconfig:
#     train_data_path = str=os.path.join("artifacts","train.csv")
#     test_data_path = str=os.path.join("artifacts","test.csv")
#     raw_data_path = str=os.path.join("artifacts","data.csv")

# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config = DataIngestionsconfig()
    
#     def initiate_data_ingestion(self):
#         logging.info("Starting data ingestion")
#         try:
#             df=pd.read_csv('nootbook\Data.csv')
#             logging.info('read the dataset as dataframe')
            
#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

#             df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
#             logging.info('saved raw data into artifacts')

#             train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

#             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

#             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
#             logging.info('split the dataset into train and test sets')

#             return{
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             }
#         except Exception as e:
#             raise CustomException(e,sys)
# /..........................................
# import os
# import sys
# import pandas as pd
# import requests
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass
# from src.exception import CustomException
# from src.logger import logging
# # from src.components.data_transformation import DataTransformation


# @dataclass
# class DataIngestionConfig:
#     """
#     Configuration for Data Ingestion.
#     """
#     raw_data_file_path: str = os.path.join("artifacts", "raw_data.csv")
#     train_data_file_path: str = os.path.join("artifacts", "train.csv")
#     test_data_file_path: str = os.path.join("artifacts", "test.csv")

# class DataIngestion:
#     """
#     Class for data ingestion.
#     """
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def fetch_data_from_binance(self, symbol='BTCUSDT', time_interval=5, limit=1000):
#         """
#         Fetch data from Binance API and return as a DataFrame.
#         """
#         try:
#             logging.info(f"Fetching data from Binance API for symbol: {symbol}, interval: {time_interval} minutes")
#             url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={time_interval}m&limit={limit}'
#             data = requests.get(url).json()

#             # Convert to DataFrame
#             D = pd.DataFrame(data)
#             D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time','qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'is_best_match']

#             logging.info("Data fetched successfully from Binance API")
#             return D

#         except Exception as e:
#             raise CustomException(e, sys)

#     def preprocess_data(self, df):
#         """
#         Preprocess the raw data.
#         - Converts timestamps to human-readable format
#         - Adds new features (month, day, year)
#         - Drops unnecessary columns
#         """
#         try:
#             logging.info("Starting data preprocessing")

#             # Convert timestamps
#             df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

#             # Feature engineering
#             df['month'] = df['open_time'].dt.month
#             df['day'] = df['open_time'].dt.day
#             df['year'] = df['open_time'].dt.year

#             # Drop unnecessary columns
#             df = df.drop(['open_time', 'close_time', 'qav', 'is_best_match'], axis=1)

#             # Ensure numerical data type for all columns
#             numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
#                                'num_trades', 'taker_base_vol', 'taker_quote_vol']
#             df[numeric_columns] = df[numeric_columns].astype(float)

#             logging.info("Data preprocessing completed")
#             return df

#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_data_ingestion(self):
#         """
#         Fetch data, preprocess it, and split it into train and test datasets.
#         """
#         try:
#             logging.info("Starting data ingestion process")

#             # Fetch raw data
#             raw_data = self.fetch_data_from_binance()

#             # Preprocess raw data
#             processed_data = self.preprocess_data(raw_data)

#             # Save raw data to CSV
#             os.makedirs(os.path.dirname(self.ingestion_config.raw_data_file_path), exist_ok=True)
#             processed_data.to_csv(self.ingestion_config.raw_data_file_path, index=False)
#             logging.info(f"Raw data saved to {self.ingestion_config.raw_data_file_path}")

#             # Split data into training and testing sets
#             train_df, test_df = train_test_split(processed_data, test_size=0.2, random_state=42)

#             # Save training and testing data to CSV
#             train_df.to_csv(self.ingestion_config.train_data_file_path, index=False)
#             test_df.to_csv(self.ingestion_config.test_data_file_path, index=False)
#             logging.info(f"Training data saved to {self.ingestion_config.train_data_file_path}")
#             logging.info(f"Testing data saved to {self.ingestion_config.test_data_file_path}")

#             return train_df, test_df

#         except Exception as e:
#             raise CustomException(e, sys)

# # /.......................................

        
# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()

# Assuming train_df and test_df are already created as DataFrames
# data_transformation = DataTransformation()
# train_array, test_array,preprocessor_path = data_transformation.initiate_data_transformation(train_df,test_df)


import os
import sys
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Configuration for Data Ingestion.
    Contains paths for saving raw, training, and testing datasets.
    """
    raw_data_file_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_file_path: str = os.path.join("artifacts", "train.csv")
    test_data_file_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    """
    Handles data fetching, preprocessing, and splitting into train and test sets.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def fetch_data_from_binance(self, symbol='BTCUSDT', time_interval=5, limit=1000):
        """
        Fetch data from the Binance API and return it as a DataFrame.

        :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
        :param time_interval: Time interval for candlestick data in minutes
        :param limit: Number of data points to fetch
        :return: DataFrame containing raw data from Binance
        """
        try:
            logging.info(f"Fetching data from Binance API for symbol: {symbol}, interval: {time_interval} minutes")

            url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={time_interval}m&limit={limit}'
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            # Convert the response to a DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'taker_base_vol',
                'taker_quote_vol', 'is_best_match'
            ])

            logging.info("Data fetched successfully from Binance API")
            return df

        except requests.exceptions.RequestException as req_err:
            raise CustomException(f"Request error: {req_err}", sys) from req_err
        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_data(self, df):
        """
        Preprocess the raw data from Binance API.

        - Converts timestamps to human-readable format
        - Adds new date-related features (month, day, year)
        - Drops unnecessary columns
        - Ensures numeric data types for relevant columns

        :param df: Raw DataFrame from Binance
        :return: Preprocessed DataFrame
        """
        try:
            logging.info("Starting data preprocessing")

            # Convert timestamps to datetime format
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

            # Add date-related features
            df['month'] = df['open_time'].dt.month
            df['day'] = df['open_time'].dt.day
            df['year'] = df['open_time'].dt.year

            # Drop unnecessary columns
            df = df.drop(['open_time', 'close_time', 'qav', 'is_best_match'], axis=1)

            # Ensure numeric data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                               'num_trades', 'taker_base_vol', 'taker_quote_vol']
            df[numeric_columns] = df[numeric_columns].astype(float)

            logging.info("Data preprocessing completed")
            return df

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self):
        """
        Fetch raw data, preprocess it, and split it into train and test datasets.

        :return: Tuple of training and testing DataFrames
        """
        try:
            logging.info("Starting data ingestion process")

            # Fetch raw data from Binance
            raw_data = self.fetch_data_from_binance()

            # Preprocess raw data
            processed_data = self.preprocess_data(raw_data)

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_file_path), exist_ok=True)

            # Save raw data to CSV
            processed_data.to_csv(self.ingestion_config.raw_data_file_path, index=False)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_file_path}")

            # Split the data into train and test sets
            train_df, test_df = train_test_split(processed_data, test_size=0.2, random_state=42)

            # Save train and test sets to CSV files
            train_df.to_csv(self.ingestion_config.train_data_file_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_file_path, index=False)
            logging.info(f"Training data saved to {self.ingestion_config.train_data_file_path}")
            logging.info(f"Testing data saved to {self.ingestion_config.test_data_file_path}")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    print("Data ingestion completed. Train and test datasets are ready.")
