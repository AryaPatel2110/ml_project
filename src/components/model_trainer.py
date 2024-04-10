import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,model_report

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts/pkl','rf.pkl')
    model_report_path = os.path.join('artifacts/model_report','rf_report.txt')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Spliting training and test input data')
            x_train,y_train,x_test,y_test  = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1], 
            )
            
            logging.info("Model Trainng Is Started")
            rf_classifier = RandomForestClassifier()
            rf_classifier.fit(x_train,y_train)
            
            logging.info("Model Is Predection Over x_test Data")
            y_pred = rf_classifier.predict(x_test)
            
            
            logging.info("Classification Report Is Initiated")
            model_report(y_pred=y_pred, y_test=y_test,file_path=self.model_trainer_config.model_report_path)
            
        except Exception as e:
            raise CustomException(e,sys)        