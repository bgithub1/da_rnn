## da_rnn_from_csv 

This is a version of Chandler Zuo's implementation of the paper:  
[*A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*](https://arxiv.org/pdf/1704.02971.pdf)  
which he details in his blog post  
[*A PyTorch Example to Use RNN for Financial Prediction*](http://chandlerzuo.github.io/blog/2017/11/darnn)

The implementation does the following:
* The model can accept input data from other times series csvs.  
  Change the input csv file by changing the first 2 lines of the code block in **section 4.01**. 

  ```
        if __name__=='__main__':
           fname_no_ext = 'nasdaq100_padding'
           rows_to_use = 5000
           ...
  ```

* The ipynb notebook can be saved as a python py file, that can be imported
   into other ipynb notebooks or other python projects.  
   Use the bash command:  
     ```jupyter nbconvert da_rnn_from_csv.ipynb --to python ```


* The da_rnn class can be saved using pickle:  
  ``` pickle.dump(m,open(f'{fname_no_ext}.pt','wb')) ```  
  Then reload it and use it to make more predictions:  
  ``` m_from_pickle = pickle.load(open(f'{fname_no_ext}.pt','wb')) ```
  
* See the ipynb notebooks **use_da_rnn_from_csv_module.ipynb** and **sine_curve_test.ipynb** to use the da_rnn model class from other ipynb notebooks.

## Use:  
To use this ipynb notebook, run all of the notebook's cells.  
In **section 4.0**, you can change the data from <span style="color:blue">nasdaq100</span> component data to <span style="color:blue">uso</span> 1-minute time bars.

The cell with the ```if __name__=='__main___``` code (in **section 4.0**) will launch:  
* Instantiating the model,
* Training of the model,
* Saving of the model,
* Reloading the model,
* Rerunning predictions using the saved model


## Data:
The folder **data** contains several csv files:
* <span style="color:blue">nasdaq100_padding.csv</span> - original csv file used by Chandler Zuo of NDX components
* <span style="color:blue">uso_full.csv</span> - one minute bar data for the commodity ETF USO
* <span style="color:blue">uso_201812.csv</span> - one minute bar data just for December 2018

You can train the model using either nasdaq100_padding.csv, uso_full.csv.  You can also use another time series csv with numeric columns, one of which is the label for that row (like other year, month, day, hour, minute open, high, low, close bar data).


## Structure of the csv/DataFrame:
The data in the csv/DataFrame can contain rows like:  
1. the returns data for the components of an index like NDX (as in Chandler Zuo's version) as in the csv file ./data/nasdaq100_padding.csv, or 
2. intra-day bar data with columns like **year, month, day, hour, minute open, high, low, close**, as in the csv file ./data/uso_full.csv


## The main() method in section 3.0
The actual model training takes place in the **main()** method in the cell below heading **3.01**, with the code:

```
    logger = setup_log()
    m = da_rnn(df_partial, logger = logger, parallel = False, learning_rate = .001)
    m.execute_training(n_epochs=100)
```

  


## Using pdb 
To use the pdb debuger, add pdb.set_trace() statements to the code.  
[See this cheatsheet for a quick reference to pdb commands](https://appletree.or.kr/quick_reference_cards/Python/Python%20Debugger%20Cheatsheet.pdf)
  