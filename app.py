from flask import Flask, jsonify,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from approval.pipeline.approval_pridict_pipeline import ApproveCustomData, ApprovePredictPipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import plotly.graph_objs as go
import json

data = pd.read_csv('notebook/final_data.csv')


application = Flask(__name__)
app = application

#  Route For A Home Page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            person_age=request.form.get('person_age'),
            person_income=request.form.get('person_income'),
            person_home_ownership=request.form.get('person_home_ownership'),
            person_emp_length=request.form.get('person_emp_length'),
            loan_intent=request.form.get('loan_intent'),
            loan_grade=request.form.get('loan_grade'),
            loan_amnt=request.form.get('loan_amnt'),
            loan_int_rate=request.form.get('loan_int_rate'),
            loan_percent_income=request.form.get('loan_percent_income'),
            cb_person_default_on_file=request.form.get('cb_person_default_on_file'),
            cb_person_cred_hist_length=request.form.get('cb_person_cred_hist_length')
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
         
        return render_template('home.html', results = results[0])
    

@app.route('/approval',methods=['GET','POST'])
def approval_predict_datapoint():
    if request.method=='GET':
        return render_template('approval.html')
    else:
        data = ApproveCustomData(
            no_of_dependents = request.form.get('no_of_dependents'),
            education = request.form.get('education'),
            self_employed = request.form.get('self_employed'),
            income_annum = request.form.get('income_annum'),
            loan_amount = request.form.get('loan_amount'),
            loan_term = request.form.get('loan_term'),
            cibil_score = request.form.get('cibil_score'),
            residential_assets_value = request.form.get('residential_assets_value'),
            commercial_assets_value = request.form.get('commercial_assets_value'),
            luxury_assets_value = request.form.get('luxury_assets_value'),
            bank_asset_value = request.form.get('bank_asset_value')
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = ApprovePredictPipeline()
        results = predict_pipeline.predict(pred_df)
         
        return render_template('approval.html', results = results[0])
    
    

@app.route('/dashboard')
def dash():
    return render_template('dash.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)