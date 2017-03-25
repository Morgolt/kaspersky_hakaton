import pandas as pd
from sklearn.preprocessing import scale

import utils
import numpy as np


def correlation(dataset):
    print(dataset.corr().head())


def make_correlation_free_set(dataset):
    dataset.drop(['tag07'], axis=1, inplace=True)
    dataset.to_csv('data/train_reduced.csv', mode='w+')


def z_normalize(dataset):
    #time = dataset['Time']
    #dataset.drop('Time', axis=1, inplace=True)
    scaled = scale(dataset)
    ds = pd.DataFrame(scaled[0:, 0:], columns=list(dataset), index=dataset.index)
    ds.to_csv('data/train_normalized.csv')


if __name__ == '__main__':
    data = utils.parse_train_data()
    print(data)
    z_normalize(data)
