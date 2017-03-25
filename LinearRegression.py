import utils
from AbstractModel import AbstractModel
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import datasets, linear_model


class LinearRegressionModel(AbstractModel):
    def __init__(self):
        self.dataset = utils.parse_train_data()

    def train(self):
        coef = np.polyfit(self.dataset.index.values, self.dataset.values, 1)
        x = self.dataset.index.values
        y = []
        for i in range(0, coef.shape[1]):
            y.append(coef[0][i] * x + coef[1][i])

        self.cls = y
        # print(coef)
        # print(y)
        # tag00 = self.dataset['tag00']
        # plt.scatter(tag00.index.values, tag00.values, color='black')
        # plt.plot(x, y[0], color='blue', linewidth=3)
        # plt.show()

        from sklearn.externals import joblib
        joblib.dump(self.cls, 'models/linear_regression.pkl')


if __name__ == '__main__':
    #cls = LinearRegressionModel()
    #cls.train()
    from sklearn.externals import joblib
    y = joblib.load('models/linear_regression.pkl')
