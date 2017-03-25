import utils
import numpy as np


def correlation(dataset):
    print(dataset.corr().head())


def make_correlation_free_set(dataset):
    dataset.drop(['tag07'], axis=1, inplace=True)
    dataset.to_csv('data/train_reduced.csv', index=False, mode='w+', )


if __name__ == '__main__':
    data = utils.parse_train_data()
    make_correlation_free_set(data)
    correlation(data)
