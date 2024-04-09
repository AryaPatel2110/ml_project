import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object, BinaryEncoder, FrequencyEncoder, OrdinalEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/pkl',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['person_age', 
                                 'person_income',
                                 'person_emp_length', 
                                 'loan_amnt',
                                 'loan_int_rate', 
                                 'loan_percent_income',
                                 'cb_person_cred_hist_length']
            
            categorical_columns = ['person_home_ownership',
                                   'loan_intent', 
                                   'loan_grade',
                                   'cb_person_default_on_file']

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_transformers = []
            for column_name in categorical_columns:
                if column_name == 'person_home_ownership' or column_name == 'loan_intent':
                    cat_transformers.append((column_name + '_frequency_encoder', FrequencyEncoder(column_name), [column_name]))
                elif column_name == 'loan_grade':
                    cat_transformers.append((column_name + '_ordinal_encoder', OrdinalEncoder(column_name), [column_name]))
                elif column_name == 'cb_person_default_on_file':
                    cat_transformers.append((column_name + '_binary_encoder', BinaryEncoder(column_name), [column_name]))


            preprocessor = ColumnTransformer(
                transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                *cat_transformers
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "loan_status"
            numerical_columns = ['person_age', 
                                 'person_income',
                                 'person_emp_length', 
                                 'loan_amnt',
                                 'loan_int_rate', 
                                 'loan_percent_income',
                                 'cb_person_cred_hist_length']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


