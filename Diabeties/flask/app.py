import pandas as pd
import numpy as np
import pickle
from flask import Flask,render_template,request,jsonify

app = Flask(__name__)
model = pickle.load(open("svc_model.pkl",'rb'))

dataset = pd.read_csv('diabetes.csv')
dataset_x = dataset.iloc[:,[1,2,5,7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset_scaled = sc.fit_transform(dataset_x)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=['POST'])
def prediction():
    features = [float(x) for x in request.form.values()]
    final_feature =[np.array(features)]
    prediction=model.predict(sc.transform(final_feature))
    
    if prediction == 1:
        pred =" You most likely to have diabetes"
    elif prediction ==0:
        pred =" You are perfecty normal "
        
    output = pred
    return render_template("index.html",prediction_text='{}'.format(output) )

if __name__ =="__main__":
    app.run(debug=True,use_reloader=False)