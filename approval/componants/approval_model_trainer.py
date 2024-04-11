import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,model_report

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts/approval/pkl','rf_approval.pkl')
    model_report_path = os.path.join('artifacts/model_report','approval_model_report.txt')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self):
        try:
            logging.info('Spliting training and test input data')
            df = pd.read_csv("D:/project/ml_project/notebook/approval/loan_approval_dataset.csv")
            x = df.drop(["loan_status"],axis=1)
            y = df["loan_status"]
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
            
            
            logging.info("Model Trainng Is Started")
            rf_classifier = RandomForestClassifier()
            
            
        
            
            rf_classifier.fit(x_train,y_train)
            
            logging.info("Model Is Predection Over x_test Data")
            y_pred = rf_classifier.predict(x_test)
            
            logging.info("Classification Report Is Initiated")
            model_report(y_pred=y_pred, y_test=y_test,file_path=self.model_trainer_config.model_report_path)
            
            logging.info("Save rf model")
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=rf_classifier
            )
            
        except Exception as e:
            raise CustomException(e,sys)        