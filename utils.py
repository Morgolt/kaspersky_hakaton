import csv
import os

import pandas as pd

import normalization


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
        print(tst_file)
        test_data = pd.read_csv(os.path.join(path, tst_file), index_col=0)
        res = det_model.test(test_data)
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


def fix_test(path='data/test'):
    tests = sorted(os.listdir(path))
    right_order = ['tag00', 'tag01', 'tag04', 'tag05', 'tag06', 'tag07', 'tag08', 'tag09', 'tag10', 'tag11', 'tag12',
                   'tag13', 'tag02', 'tag15', 'tag16', 'tag17', 'tag18']
    scaler = normalization.get_scaler()
    for testfile in tests:
        print(testfile)
        test_data = pd.read_csv(os.path.join(path, testfile), index_col=0)
        test_data = test_data[right_order]
        test_data.fillna(method='pad', inplace=True)
        test_data.to_csv(os.path.join('data/test_fixed', testfile))
        scaled = scaler.transform(test_data)
        scaled = pd.DataFrame(scaled, columns=right_order, index=test_data.index)
        scaled.to_csv(os.path.join('data/test_norm', testfile))



if __name__ == '__main__':
    # parse_data('data/train.csv')
    #model = AbstractModel()
    #test_model(model, 'data/test')
    # mean_all_answers()
    fix_test()