# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:53:48 2021

@author: rahul
"""


#This program is used to predict the weather data based on the data on the file temeperature.csv using the ARIMA model
#for weather prediction. The model will take an aprroximate 3 hours to run but I have saved the predicted and expected 
#values of the Model in the "Pred_Test.csv" file

# import modules
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from tqdm import tqdm

import statsmodels.tsa.api as smt


# training the data using ARIMA model 
def train():
    data = pd.read_csv('temperatures.csv', index_col=['Zeitstempel'], parse_dates=['Zeitstempel'])
    data=data['Wert']
    X = data.values
    size = int(len(X) * 0.90)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in tqdm(range(len(test))):
        model = ARIMA(history, order=(1,1,1))
        

        model_fit = model.fit()
        
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    #print(predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    #print(test)
    test=list(test)
    d = {'Expected':test , 'Predicted':predictions }
    o = pd.DataFrame(data=d)
    #print(o)
    o.to_csv('Pred_Test.csv')
#Plot a graph between Expected and Predicted
    plt.plot(test,color='blue',label='Excepted')
    plt.plot(predictions, color='red',label='Predicted')
    plt.legend()
    plt.savefig("metric.png")
    plt.clf()

# Find the residual of the Model
    resid = model_fit.resid
    stats.normaltest(resid)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    plt.savefig("qqplot.png")
    plt.clf()

    print(model_fit.params)
    print(model_fit.aic, model_fit.bic, model_fit.hqic)

if __name__ == '__main__':
    train()