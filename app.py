from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('dpm.pickle', 'rb'))

def lrprob(lr, x):
    co = lr.coef_[0]
    log_odd = co[0] * x[0] + co[1] * x[1] + co[2] * x[2] + co[3] * x[3] + co[4] * x[4] + co[5] * x[5] + lr.intercept_[0]
    odds = np.exp(log_odd)
    prob = odds / (1 + odds)
    prob = prob * 100
    prob = round(prob, 3)
    prob = str(prob)+'%'
    return (prob)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predection', methods=['GET', 'POST'])
def predection():
    if request.method == 'POST':
        try:
            X = [int(request.values.get('bin')), int(request.values.get('age')), int(request.values.get('ht')), int(request.values.get('hd')), int(request.values.get('bmi')), int(request.values.get('bml'))]
        except TypeError:
            return redirect(url_for('home'))
        pred = lrprob(model, X)
        return render_template('index.html', predection=pred)
    return render_template('index.html', predection="Form Failed, Please Try Again")
    
    
if __name__ == '__main__':
    app.run(debug=True)
