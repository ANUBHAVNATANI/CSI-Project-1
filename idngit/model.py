#Code by Anubhav Natani
#--------------------------------#
#importing libraries
import numpy as np
import pandas as pd
import sklearn
import requests,json
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
"""
i have already cleaned the dataset made it for prediction
more can be seen in the ipython notebook 
i am saving the dataset in the another file and use that for prediction
"""
X_train=pd.read_csv('train.csv',index_col=0)
Y_train=pd.read_csv('train_y.csv',index_col=0)
X_test=pd.read_csv('test.csv',index_col=0)
Y_test=pd.read_csv('test_y.csv',index_col=0)
"""
I have already tried all the best models and selected the model that way all can be seen in the
ipython notebook.
"""

lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
"""
As i want my model to once in for all trained and it need not to
be trained on the same data again and again so i am using pickling to save the instance in bytecode.
"""
BASE_URL = "http://localhost:5000"

joblib.dump(lin_reg,'model.pkl')
joblib.dump(X_train,"training_data.pkl")
joblib.dump(Y_train,"training_labels.pkl")

#testing model
"""
loaded_model = pickle.load(open(filename,'rb'))
result = loaded_model.score(X_test,Y_test)
print(result)
"""
#Predict API
datas = {"Mid_Term_Agg":16.67,"Lab":23,"Quiz1":7.5,"Branch_cov":4,"CP_grade_cov":5,"CP_Lab_grade_cov":7}
response = requests.post("{}/predict".format(BASE_URL),json=datas)
response.json()
"""
As the new data arrives i can train the model once again using the new features and the data.
"""
##that part i will do later
