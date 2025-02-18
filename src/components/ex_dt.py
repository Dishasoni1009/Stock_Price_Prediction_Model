# from src.components.data_transformation import DataTransformation

# # Assuming train_df and test_df are already created as DataFrames
# data_transformation = DataTransformation()
# train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_df, test_df)

# # Separate input features and target
# X_train, y_train = train_array[:, :-1], train_array[:, -1]
# X_test, y_test = test_array[:, :-1], test_array[:, -1]

# # Train model
# RF = RandomForestRegressor(random_state=42)
# RF.fit(X_train, y_train)

# # Evaluate
# y_pred = RF.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print('Mean Squared Error:', mse)


# from src.components.data_ingestion import DataIngestion

# # Instantiate Data Ingestion
# data_ingestion = DataIngestion()

# # Initiate Data Ingestion
# train_df, test_df = data_ingestion.initiate_data_ingestion()

# # Output the first few rows of the training data
# print(train_df.head())
# print(test_df.head())


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

def main():
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_df, test_df = data_ingestion.initiate_data_ingestion()
        
        print("Train DataFrame Sample:")
        print(train_df.head())
        
        print("Test DataFrame Sample:")
        print(test_df.head())
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_df, test_df)
        
        print(f"Train array shape: {train_array.shape}")
        print(f"Test array shape: {test_array.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
