from sklearn import svm

import utils
from AbstractModel import AbstractModel


class OneClassSvmModel(AbstractModel):
    def __init__(self):
        self.dataset = utils.parse_train_data()
        self.dataset.drop('Time', axis=1, inplace=True)
        self.cls = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)

    def test(self, dataset):
        pass

    def train(self):
        self.cls.fit(self.dataset)
        from sklearn.externals import joblib
        joblib.dump(self.cls, 'models/one_class_svm.pkl')


if __name__ == '__main__':
    cls = OneClassSvmModel()
    cls.train()

