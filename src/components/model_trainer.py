import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
            
            param = {
                'bootstrap': [True],
                'max_depth': [80,90,100,None],
                'n_estimators': [400,600,800],
                }
            gs = GridSearchCV(rf_classifier,param_grid=param,cv=3)
            
            logging.info("Model Tuning Started")
            gs.fit(x_train,y_train)
            rf_classifier.set_params(**gs.best_params_)
            print(gs.best_params_)
            
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