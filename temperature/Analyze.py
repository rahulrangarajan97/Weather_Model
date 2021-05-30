# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:53:48 2021

@author: rahul
"""

# import modules
import pandas as pd
import numpy as np
from pandas import DataFrame as da
from datetime import datetime as dt
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from itertools import product
from pylab import rcParams

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt

# The Analysis() funtion is used to Store the hottest and coldest temperatures of each year in a csv file and plot the same
def Analysis():

    # Read Dataframe and upsample the data to 15min intervals
    df=pd.read_csv('temperatures.csv')
    df['Zeitstempel']=pd.to_datetime(df['Zeitstempel'],format='%Y%m%d%H%M')
    df.index = df['Zeitstempel']
    a=pd.DataFrame(df.Wert)
    df=a.resample('15min').interpolate(method='cubic')


    # A plot of the Data w.r.t. Date Time

    plt.plot(df.index,df.Wert)
    plt.title("DateTime vs Temperature")
    plt.xlabel("DateTime")
    plt.ylabel("Temperature")
    plt.savefig("DateTime vs Temperature.png")
    plt.clf()
    
    years=sorted(list(set(df.index.year)))

    max_temp = pd.DataFrame()
    min_temp = pd.DataFrame()
    
#Loop for isolating the maximimum and minimum temperatures of each year directly from the Data Frame

    for y in years:
        max_temp = max_temp.append(df.loc[[df[str(y)][df.columns[0]].idxmax()]])
        min_temp = min_temp.append(df.loc[[df[str(y)][df.columns[0]].idxmin()]])
        
# Saving the Max and Min Temp of each Year to a CSV file
    e=list(max_temp.index)
    f=list(max_temp.Wert)
    g={'time':e,'Wert':f}
    max_temp=pd.DataFrame(g,columns=['time','Wert'])
    #print(max_temp)
    e=list(min_temp.index)
    f=list(min_temp.Wert)
    g={'time':e,'Wert':f}
    min_temp = pd.DataFrame(g,columns=['time','Wert'])
    #print(min_temp)
    o=pd.concat([max_temp,min_temp],axis=1)
    o.columns=['time','maxtemp','time','mintemp']
    #print(o)
    
    o.to_csv('Max and Min Temp of each Year.csv')

    max_temp['year'] = [i.year for i in max_temp.time]
    min_temp['year'] = [i.year for i in min_temp.time]
    max_temp['month'] = [i.strftime('%Y-%m') for i in max_temp.time]
    min_temp['month'] = [i.strftime('%Y-%m') for i in min_temp.time]
    
#Plotting the Maximum and Minimum temeperatures of each year
    plt.plot(max_temp.month,max_temp.Wert)
    plt.title("MaxTemp per Year")
    plt.xlabel("DateTime")
    plt.ylabel("Temperture")
    plt.savefig("MaxTemp per Year.png")
    plt.clf()

    plt.plot(min_temp.month,min_temp.Wert)
    plt.title("MinTemp per Year")
    plt.xlabel("DateTime")
    plt.ylabel("Temperture")
    plt.savefig("MinTemp per Year.png")
    plt.clf()
 # A two side plot to visualize the max and min temeperatures on the same Plot   
    x = df.index
    y1=df.Wert
    fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
    plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
    plt.ylim(-800, 800)
    plt.title('Weather Data (Two Side View)', fontsize=16)
    plt.xlabel("DateTime")
    plt.ylabel("Temperture")
    plt.hlines(y=0, xmin=np.min(df.index), xmax=np.max(df.index), linewidth=.5)
    #plt.show()
    plt.savefig('Weather Data (Two Side View).png')
    plt.clf()
# A plot to view the avg. temeperature per month and per year respectively
    df['year'] = df.index.year
    #by year
    df_avg_year = df.groupby(df.year).mean()['Wert']
    plt.plot(df_avg_year)
    plt.title("Avg Temp per Year")
    plt.xlabel("DateTime")
    plt.ylabel("Temperture")
    plt.savefig('Avg Temp per Year.png')
    plt.clf()
    df['month'] = df.index.month
    
    #by month
    df_avg_year = df.groupby(df.month).mean()['Wert']
    plt.plot(df_avg_year)
    plt.title("Avg Temp per Month")
    plt.xlabel("DateTime")
    plt.ylabel("Temperture")
    plt.savefig("Avg Temp per Month.png")
    plt.clf()
    
# A box Plot to visualize the trend and the anomaly temperatures of each year
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x='year', y='Wert', data=df, ax=axes[0])
    sns.boxplot(x='month', y='Wert', data=df)

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    #plt.show()
    plt.savefig("Box Plot.png")
    plt.clf()


    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(df.Wert, model='additive',period=365)
    fig = decomposition.plot()
    #plt.show()
    plt.savefig('DecompositionPlot.png')
    plt.clf()
    
# Bonus: A function to plot the moving average of the data
    def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

        rolling_mean = series.rolling(window=window).mean()

        plt.figure(figsize=(17,8))
        plt.title('Moving average\n window size = {}'.format(window))
        plt.plot(rolling_mean, 'g', label='Rolling mean trend')

        #Plot confidence intervals for smoothed values
        if plot_intervals:
                mae = mean_absolute_error(series[window:], rolling_mean[window:])
                deviation = np.std(series[window:] - rolling_mean[window:])
                lower_bound = rolling_mean - (mae + scale * deviation)
                upper_bound = rolling_mean + (mae + scale * deviation)
                plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
                plt.plot(lower_bound, 'r--')
                
        plt.plot(series[window:], label='Actual values')
        plt.legend(loc='best')
        plt.grid(True)
# Plot the moving average by the previous 365 days(by week) and by previous quarter(90 days)        
    data = pd.read_csv('temperatures.csv', index_col=['Zeitstempel'], parse_dates=['Zeitstempel'])
    data.head(10)
    data=data['Wert']
    #Smooth by the previous 365 days (by week)
    plot_moving_average(data, 365)
    plt.savefig("Mv_365.png")
    plt.clf()
    # #Smooth by previous quarter (90 days)
    plot_moving_average(data, 90, plot_intervals=True)
    plt.savefig("Mv_90.png")
    plt.clf()

if __name__ == '__main__':
    Analysis()