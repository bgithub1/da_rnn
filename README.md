### This is a version of Chandler Zuo's implementation of the paper:

*A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction (see http://chandlerzuo.github.io/blog/2017/11/darnn)*

The code allows one to replace the input data with other times series csvs.

### To change the input csv file by changing the first 2 lines of the code block in **section 4.01**.  

```
    if __name__=='__main__':
        fname_no_ext = 'nasdaq100_padding' 
        rows_to_use = 5000
        ...
```

### The data in the DataFrame can be like:
   1. the returns data for the components of an index like NDX (as in Chandler Zuo's version) as in the csv file ./data/nasdaq100_padding.csv, or 
   2. intra-day bar data with columns like **year, month, day, hour, minute open, high, low, close**, as in the csv file ./data/uso_full.csv

### The actual model training takes place in the **main()** method in the cell below heading **3.01**, with the code:

```
    logger = setup_log()
    m = da_rnn(df_partial, logger = logger, parallel = False, learning_rate = .001)
    m.execute_training(n_epochs=100)
```

  


### To use the pdb debuger, add pdb.set_trace() statements to the code.

[See the cheatsheet:  https://appletree.or.kr/quick_reference_cards/Python/Python%20Debugger%20Cheatsheet.pdf ]
