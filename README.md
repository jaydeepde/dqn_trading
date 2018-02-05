# Deep Q Learning for Trading
## Problem Statement :
This is demo to use Deep Q Learning for Systamic Trading. In this demo the data is for predicting the oil prices and taking action (1-12 buy and hold, 1-12 hours of short and hold, 1-12 hours of hold) based on past 12 hours of price data at 4 secs interval.
## how to run dqn_trading?
```bash
python run_trading_1d.py
```
Data is inside data/*
Results will be saved in Trading_Experiment_*/results.csv
Learning rate and loss will be stored in learning.csv file
After each iteration the network parameters will be stored
## How to analyze results?
Open and run the ipython notebook analyse_results.ipynb

## Future work
1. Implement for more applications for time series predictions.
2. Implement new architechtures like Double Q Learning, Dueling network etc.
3. Implement new ideas.

## Some starting materials 
1. The slides attached RL_presentation.pptx
2. [Demystifying Deep Q Learning](https://ai.intel.com/demystifying-deep-reinforcement-learning/)
