from flask import Flask,jsonify,request
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

app=Flask(__name__)

@app.route("/predict",methods=['POST'])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()
            datas=float(zip(data["Mid_Term_Agg"],data["Lab"],data["Quiz1"],data["Branch_cov"],data["CP_grade_cov"],data["CP_Lab_grade_cov"]))
            lin_reg = joblib.load("./model.pkl")
        except ValueError:
            return jsonify("Please enter a number.")
        
        return jsonify(lin_reg.predict(datas).tolist)

if __name__ =='main':
    app.run(debug=True)