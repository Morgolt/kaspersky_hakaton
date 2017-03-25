import csv
import os

import pandas as pd

from AbstractModel import AbstractModel


def parse_train_data(path='data/train.csv'):
    """
    Read training data.
    :param path: path to train.csv
    :return: Pandas DataFrame
    """
    data = pd.read_csv(path, index_col=0)
    return data


def test_model(det_model, path='data/test'):
    """
    Writes results file under the name of model
    :param det_model: Fitted model, should implement AbstractModel
    :param path: path to test directory
    """
    result = []
    i = 0
    for tst_file in os.listdir(path):
        print("testfile")
        test_data = parse_train_data(os.path.join(path, tst_file))
        result.append((i, det_model.test(test_data),))
        i += 1

    with open('output/{0}.csv'.format(det_model), 'w+') as output:
        writer = csv.writer(output, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(result)


if __name__ == '__main__':
    # parse_data('data/train.csv')
    model = AbstractModel()
    test_model(model, 'data/test')
