from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
      obj = DataIngestion()
      train_data, test_data, _ = obj.initiate_data_ingestion()
      data_transformation = DataTransformation()
      train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)
      
      model_trainner = ModelTrainer()
      model_trainner.initiate_model_trainer(test_array=test_arr,train_array=train_arr)