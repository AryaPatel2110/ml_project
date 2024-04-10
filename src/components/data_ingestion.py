import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass




@dataclass
class DaataIngestionConfig:
    train_data_path = os.path.join('artifacts/data','train.csv')
    test_data_path = os.path.join('artifacts/data','test.csv')
    raw_data_path = os.path.join('artifacts/data','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DaataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Enter The Data Ingestion Method or Component")
        try:
            df = pd.read_csv('notebook/final_data.csv')
            logging.info('Read The Dataset as Dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)             
            
            logging.info('Ingestion of Data Is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        