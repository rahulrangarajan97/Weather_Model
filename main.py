#THIS PROGRAM IS USED TO RUN BOTH THE ANALYSIS AND THE PREDICTION PART OF THE DATA.THE PREDICTION MODEL 'arima.train()' 
#HAS BEEN COMMENTED AS THE MODEL WILL TAKE AN APPROXIMATE 3hrs TO RUN.BUT I HAVE SAVED THE PREDICTED AND EXPECTED VALUES
#OF THE MODEL IN "Pred_Test.csv" FILE




from temperature import Analyze
from temperature import arima










if __name__ == '__main__':
	Analyze.Analysis()
	#arima.train()