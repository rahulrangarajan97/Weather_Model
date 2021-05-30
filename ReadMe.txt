This is a program to do a analysis on the data file Temperatures.csv

The program consists of three python files- main.py , (Analyze.py , arima.py)-inside the folder temperature

The program main.py is the main program used to run the two sub programs Analyze and arima

The program file analyze.py is used to upsample the data to 15minute intervals and find the max and min temp of each year and plot them respectively

The program file Arima.py consists of the ARIMA model used to predict the weather based on the given data

NOTE : The function arima.train()in the python file main.py has been commented as the model will take an approximate 3hrs to train and predict the data. The trained and predicted data found already while running the model have been saved in the excel file 'Pred_test.csv'

PLOT FILES:
1.Datetime vs Temperature.png- A plot of the datatime against the temperature of the given data
2.decomposition plot.png - A plot consisting of the seasonal and residual plots of the data
3.Maxtemp per year.png- A plot of the max temps of each year against their time of occurence
4.Mintemp per year.png- A plot of the min temps of each year against their time of occurence
5.Mv_90.png- A moving average plot for quarterly time period of the data
6.Mv_360.png- A moving average plot for the 365 days(by week) of the data
7.Weatherdata(two side view).png - A plot of the max and min temperatures over the years 
8.metric.png - A plot of the predicted values against the Expected values obtained while running the ARIMA model
9.qqplot.png - A plot of the residual of the ARIMA model

EXCEL FILES:
1.temperatures.csv - The actual data provided for the programming task
2.Max and Min temp of each year.csv - A dataframe consisting of the max and min temp of each year and their respective time of occurance
3.Pred_test.csv- A dataframe consisting of the predicted and expected values of the data obtained while running the ARIMA model