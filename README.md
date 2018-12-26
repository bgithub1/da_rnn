This version of Chandler Zuo's implementation of the paper: **A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction** (see http://chandlerzuo.github.io/blog/2017/11/darnn)
generalizes the input data to Chandler Zuo's da_rnn model.  

To use:

Instantiate the class da_rnn with a pandas DataFrame whose columns are numeric features of time series data (NOT the path to a csv file).  The data can be like:
  1. the returns data for the components of an index like NDX (as in Chandler Zuo's version) as in the csv file ./data/nasdaq100_padding.csv, or 
  2. intra-day bar data with columns like **year, month, day, hour, minute open, high, low, close**, as in the csv file ./data/uso_full.csv
  
The actual model training takes place in the cell below heading **4.01**, with the code:

```   m = execute_training(df_partial,n_epochs=100,learning_rate=.001)```


