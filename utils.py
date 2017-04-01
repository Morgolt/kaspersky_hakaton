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
    tests = sorted(os.listdir(path))
    for tst_file in tests:
        test_data = pd.read_csv(os.path.join(path, tst_file), index_col=0)
        res = det_model.test(test_data)
        print(tst_file)
        print(res)
        result.append((i, res,))
        i += 1

    with open('output/{0}.csv'.format(det_model), 'w+') as output:
        writer = csv.writer(output, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(result)

def mean_all_answers():
    frames = []
    for answ_file in os.listdir('output'):
        frames.append(pd.read_csv(os.path.join('output', answ_file), index_col=0))
    concated_frame = pd.concat(frames, axis=1)
    print(concated_frame)
    result = concated_frame.mean(axis=1)
    print(result)
    result.to_csv('output/meaned.csv')

if __name__ == '__main__':
    # parse_data('data/train.csv')
    model = AbstractModel()
    test_model(model, 'data/test')
    # mean_all_answers()
