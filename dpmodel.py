import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
#from sklearn.metrics import r2_score

df = pd.read_csv('C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\Supervised ML\\Logistic Regression\\diabetes_prediction_dataset.csv')
df.dropna()

gender = {'Female':0, 'Male':1, 'Other':2}
df['gender'] = df['gender'].map(gender)

features = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'blood_glucose_level']

X = df[features]
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X,y)

with open('dpm.pickle', 'wb') as f:
    pickle.dump(lr, f)

def lrprob(lr, x):
    co = lr.coef_[0]
    log_odd = co[0] * x[0] + co[1] * x[1] + co[2] * x[2] + co[3] * x[3] + co[4] * x[4] + co[5] * x[5] + lr.intercept_[0]
    odds = np.exp(log_odd)
    prob = odds / (1 + odds)
    prob = prob * 100
    prob = round(prob, 2)
    prob = str(prob)+'%'
    return (prob)

print(lrprob(lr,[1,100,1,1,40,140]))
    
# predict_proba([[1,97,1,1,30,120]])