import pickle 
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, app, jsonify, url_for


app = Flask(__name__)

## Load the model
rf_model = pickle.load(open('rf_model.pkl', 'rb'))


@app.route('/')
def home():

    return render_template('home.html')

@app.route('/predict_api', method = ['POST'])
def predict_api():
    data = request.json['data']
    X = pd.DataFrame(data, index = [0])

    X["Self_Employed"] = np.where(X["Self_Employed"] == "Yes", 1, np.where(X["Self_Employed"].isna(), X["Self_Employed"], 0))
    X["Married"] = np.where(X["Married"] == "Yes", 1, np.where(X["Married"].isna(), X["Married"], 0))
    X["Male"] = np.where(X["Gender"] == "Male", 1, np.where(X["Gender"].isna(), X["Gender"], 0))
    X["Graduate"] = np.where(X["Education"] == "Graduate", 1, np.where(X["Education"].isna(), X["Education"], 0))

    ## Dealing with the number of dependents
    X[["1","2","3+"]] = pd.Series([0,0,0])
    dep = X["Dependents"][0]
    X[dep] = 1

    ## Dealing with the area
    X[["Semiurban","Urban"]] = pd.Series([0,0])
    ar = X["Area"][0]
    X[ar] = 1


    X.drop(["Gender", "Education","Dependents", "Area"], inplace = True, axis = 1) 


    X = X[rf_model.feature_names_in_]

    out = rf_model.predict(X)
    print(out[0])
    return jsonify(out[0])

if __name__ == "__main__":
    app.run(debug=True)
