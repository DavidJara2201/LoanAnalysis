import pickle 
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, app, jsonify, url_for


app = Flask(__name__)

## Load the model
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

## Name of the original columns 
cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Applicant_Income", "Coapplicant_Income",
        "Loan_Amount", "Term", "Area"]

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    X = pd.DataFrame(data, index = [0])


    X["Self_Employed"] = np.where(X["Self_Employed"] == "Yes", 1, np.where(X["Self_Employed"].isna(), X["Self_Employed"], 0))
    X["Married"] = np.where(X["Married"] == "Yes", 1, np.where(X["Married"].isna(), X["Married"], 0))
    X["Male"] = np.where(X["Gender"] == "Male", 1, np.where(X["Gender"].isna(), X["Gender"], 0))
    X["Graduate"] = np.where(X["Education"] == "Graduate", 1, np.where(X["Education"].isna(), X["Education"], 0))

    ## Dealing with the number of dependents
    X["1"] = 0
    X["2"] = 0
    X["3+"] = 0
    dep = X["Dependents"][0]
    X[dep] = 1

    ## Dealing with the area
    X["Semiurban"] = 0
    X["Urban"] = 0
    ar = X["Area"][0]
    X[ar] = 1


    X.drop(["Gender", "Education","Dependents", "Area"], inplace = True, axis = 1) 


    X = X[rf_model.feature_names_in_]

    out = rf_model.predict(X)
    return jsonify(out[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = []
    for var in cols:
        data.append(request.form.get(var))

    X = pd.DataFrame(columns = cols)
    X = pd.concat([X, pd.DataFrame([data], columns = cols)])


    X["Dependents"] = X["Dependents"].apply(str)
    
    X["Self_Employed"] = np.where(X["Self_Employed"] == "Yes", 1, np.where(X["Self_Employed"].isna(), X["Self_Employed"], 0))
    X["Married"] = np.where(X["Married"] == "Yes", 1, np.where(X["Married"].isna(), X["Married"], 0))
    X["Male"] = np.where(X["Gender"] == "Male", 1, np.where(X["Gender"].isna(), X["Gender"], 0))
    X["Graduate"] = np.where(X["Education"] == "Graduate", 1, np.where(X["Education"].isna(), X["Education"], 0))

    ## Dealing with the number of dependents
    X["1"] = 0
    X["2"] = 0
    X["3+"] = 0
    dep = X["Dependents"][0]
    X[dep] = 1

    ## Dealing with the area
    X["Semiurban"] = 0
    X["Urban"] = 0
    ar = X["Area"][0]
    X[ar] = 1


    X.drop(["Gender", "Education","Dependents", "Area"], inplace = True, axis = 1)

    X = X[rf_model.feature_names_in_]

    out = rf_model.predict(X)[0]
    if out == "Y":
        text = "Congratulations, you are very likely to receive a loan"
    else:
        text = "Unfortunately you do not seem to fit for a loan at this moment. However, we encourage you to contact the bank for a deeper study if you believe you should be considered"

    return render_template("home.html", prediction_text = text)


if __name__ == "__main__":
    app.run(debug=True)
