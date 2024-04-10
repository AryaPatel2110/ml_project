import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def model_report(y_pred, y_test, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            # Accuracy Score
            acc_score = accuracy_score(y_test, y_pred)
            file.write("Accuracy score of the model: {:.4f}\n".format(acc_score))
        
            # Classification report
            file.write("Classification report:\n")
            class_rep = classification_report(y_test, y_pred)
            file.write(class_rep + "\n")
        
            # Confusion Matrix
            plt.figure(figsize=(6, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Reds", fmt='g')
            plt.title('Confusion matrix: Random Forest')
            plt.savefig('artifacts/output/confusion_matrix.png')
            plt.close()
        
            file.write("Confusion matrix saved as confusion_matrix.png")
            
    except Exception as e:
        raise CustomException(e,sys)







class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        self.frequency_map = X[self.column_name].value_counts().to_dict()
        return self

    def transform(self, X):
        X[self.column_name] = X[self.column_name].map(self.frequency_map)
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.column_name] = X[self.column_name].map({'N': 0, 'Y': 1})
        return X


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        grade = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
        X[self.column_name] = X[self.column_name].map(grade)
        return X
