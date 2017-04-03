from pandas import read_csv, DataFrame
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import utils

def adf_test(dataset='data/train_normalized.csv'):
    for col_name in dataset.columns:
        print('==========================================================')
        test = sm.tsa.adfuller(dataset[col_name], maxlag=6)
        print('adf: ', test[0])
        print('p-value: ', test[1])
        print('Critical values: ', test[4])
        if test[0] > test[4]['5%']:
            print('есть единичные корни, ряд ', col_name, ' не стационарен')
        else:
            print('единичных корней нет, ряд ', col_name, 'стационарен')
        print('==========================================================')