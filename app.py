from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_fraud.pkl'
#model = pickle.load(open(filename, 'rb'))
xg = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    amount = float(request.form['amount'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    newbalanceDest = float(request.form['newbalanceDest'])

    pred = xg.predict(np.array([[oldbalanceOrg, amount, newbalanceOrig, newbalanceDest]]))
    print(pred)
    return render_template('index.html', predict=str(pred))



if __name__ == '__main__':
    app.run

