from flask import Flask,request,render_template,jsonify,url_for
import pickle
import numpy as np 
import pandas as pd
import sklearn
import requests,json
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression


def conv(a):
    if(a=="F"):
        return(0)
    elif(a=="D"):
        return(1)
    elif(a=="CD"):
        return(2)
    elif(a=="C"):
        return(3)
    elif(a=="BC"):
        return(4)
    elif(a=="B"):
        return(5)
    elif(a=="AB"):
        return(6)
    else:
        return(7)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dsa',methods=['GET'])
def dsa():
    if request.method == "GET":
        return render_template('dsa.html')


@app.route('/anel',methods=['GET'])
def anel():
    if request.method == "GET":
        return render_template('anel.html')

@app.route('/predictd',methods=["POST","GET"])
def predictd():

    load_model=open("model.pkl","rb")
    lin_reg_model=joblib.load(load_model)


    if request.method == "POST" or request.method == "GET":
        Mid_Term_Agg = request.form['Mid_Term']
        Mid_Term_Agg = float(Mid_Term_Agg)
        Mid_Term = (Mid_Term_Agg/60)*25
        Lab = request.form['Lab']
        Quiz1 = request.form['Quiz']
        Branch_cov = request.form['Branch']
        CP_grade_cov = request.form['CP_Grade']
        CP_grade_cov = conv(CP_grade_cov)
        CP_Lab_grade_cov = request.form['CP_Lab']
        CP_Lab_grade_cov = conv(CP_Lab_grade_cov)
        datas=[]
        datas.extend((Mid_Term,Lab,Quiz1,Branch_cov,CP_grade_cov,CP_Lab_grade_cov))
        datas=[float(i) for i in datas]
        datas=[datas]
        mypred=lin_reg_model.predict(datas)
        mypred=np.round(mypred)
    return render_template('resultd.html', prediction= mypred)

@app.route('/predicta',methods=["POST","GET"])
def predicta():

    load_model=open("modela.pkl","rb")
    lin_reg_model=joblib.load(load_model)


    if request.method == "POST" or request.method == "GET":
        Mid_Term = request.form['Mid_Term']
        Mid_Term = float(Mid_Term)
        Quiz1 = request.form['Quiz']
        Branch_cov = request.form['Branch']
        BE_grade_cov = request.form['BE_Grade']
        BE_grade_cov = conv(BE_grade_cov)
        datas=[]
        datas.extend((Mid_Term,Quiz1,Branch_cov,BE_grade_cov))
        datas=[float(i) for i in datas]
        datas=[datas]
        mypred=lin_reg_model.predict(datas)
        mypred=np.round(mypred)
    return render_template('resulta.html', prediction= mypred)


if __name__ == '__main__':
    app.run(debug=True)
