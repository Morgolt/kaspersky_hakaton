import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


def correlation(dataset):
    print(dataset.corr().head())


def make_correlation_free_set(dataset):
    dataset.drop(['tag07'], axis=1, inplace=True)
    dataset.to_csv('data/train_reduced.csv', mode='w+')


def normalize_train(trainset):
    scaler = StandardScaler()
    scaled_trainset = scaler.fit_transform(trainset)
    ds = pd.DataFrame(scaled_trainset, columns=list(trainset), index=trainset.index)
    ds.to_csv('data/train/normalized.csv')
    with open('models/scaler.pkl', 'wb') as scalefile:
        pickle.dump(scaler, scalefile)
    return scaler

def get_scaler():
    try:
        with open('models/scaler.pkl', 'rb') as scfile:
            scaler = pickle.load(scfile)
        return scaler
    except:
        raise IOError('Call "normalize_train(trainset) first."')


if __name__ == '__main__':
    ##data = utils.parse_train_data()
    #print(data)
    #normalize_train(data)
    scaler = get_scaler()
    print(scaler.scale_)
