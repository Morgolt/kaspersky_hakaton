import utils
from AbstractModel import AbstractModel
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import datasets, linear_model


class LinearRegressionModel(AbstractModel):
    def test(self, dataset):
        from sklearn.externals import joblib
        dataFrame = joblib.load('models/linear_regression_dataframe.pkl')

        x = dataset.index.values
        temp_list = []
        for col_name in dataFrame.index.values:
            y = dataFrame.loc[dataFrame.index == col_name]['a'][0] * x + \
                dataFrame.loc[dataFrame.index == col_name]['b'][0]
            real_y = dataset[col_name].values
            error = abs(real_y - y)
            errDF = pd.DataFrame({col_name: error}, index=x)
            temp_list.append(errDF)
        errorDF = pd.concat(temp_list, axis=1)
        result = errorDF.sum(axis=1).idxmax()
        print("error: ", result)
        return result

    def __init__(self):
        self.dataset = utils.parse_train_data()

    def train(self):
        coef = np.polyfit(self.dataset.index.values, self.dataset.values, 1)
        self.dataFrame = pd.DataFrame(coef.transpose(), index=self.dataset.columns, columns=['a', 'b'])
        print(self.dataFrame)
        # print(coef)
        # print(y)
        # tag00 = self.dataset['tag00']
        # plt.scatter(tag00.index.values, tag00.values, color='black')
        # plt.plot(x, y[0], color='blue', linewidth=3)
        # plt.show()

        from sklearn.externals import joblib
        joblib.dump(self.dataFrame, 'models/linear_regression_dataframe.pkl')


if __name__ == '__main__':
    #cls = LinearRegressionModel()
    #cls.train()
    #from sklearn.externals import joblib
    #y = joblib.load('models/linear_regression.pkl')
    model = LinearRegressionModel()
    utils.test_model(model)



