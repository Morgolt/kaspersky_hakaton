import numpy
from sklearn import svm
from sklearn.externals import joblib

import utils
from AbstractModel import AbstractModel


class OneClassSvmModel(AbstractModel):
    def __init__(self):
        self.dataset = utils.parse_train_data(path='data/train_normalized.csv')
        # self.dataset.drop('Time', axis=1, inplace=True)
        self.cls = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.1)

    def test(self, dataset):
        self.trained = joblib.load('models/one_class_svm_norm.pkl')
        result = self.trained.predict(dataset)
        res_ind = numpy.argmax(result)
        print(res_ind)
        return dataset.index.values[res_ind]

    def train(self):
        self.cls.fit(self.dataset)
        joblib.dump(self.cls, 'models/one_class_svm_norm.pkl')


if __name__ == '__main__':
    cls = OneClassSvmModel()
    utils.test_model(cls)
