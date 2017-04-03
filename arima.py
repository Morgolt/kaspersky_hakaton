import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import utils
from AbstractModel import AbstractModel
from statsmodels.tsa.arima_model import ARIMA


def calculate_q_param(timeseries, alpha=0.1, fft=True):
    # timeseries is dataset[column_name]
    acf = sm.tsa.stattools.acf(timeseries, fft=fft,
                               nlags=timeseries.__len__())
    return np.argmin(acf > alpha)


def calculate_p_param(timeseries, alpha=0.1, nlags=100):
    pacf = sm.tsa.stattools.pacf(timeseries, nlags=nlags)
    return np.argmin(pacf > alpha)


def q_params(dataset):
    list = []
    for col_name in dataset.columns:
        print('==========================================================')
        print(col_name)
        q = calculate_q_param(dataset[col_name])
        print('q: ', q)
        list.append(q)
        print('==========================================================')
    q_paramsDF = pd.DataFrame(list, index=dataset.columns, columns=['q'])
    return q_paramsDF


def p_params(dataset):
    list = []
    for col_name in dataset.columns:
        print('==========================================================')
        print(col_name)
        p = calculate_p_param(dataset[col_name])
        print('p: ', p)
        list.append(p)
        print('==========================================================')
    p_paramsDF = pd.DataFrame(list, index=dataset.columns, columns=['p'])
    return p_paramsDF


def save_qparams(qDF, filename='models/arima_q_paramsDF.pkl'):
    joblib.dump(qDF, filename)


def save_pparams(pDF, filename='models/arima_p_paramsDF.pkl'):
    joblib.dump(pDF, filename)


class ArimaModel(AbstractModel):
    #def test(self, dataset):
    def __init__(self):
        self.dataset_date = utils.parse_train_data('data/train_normalized.csv')
        #self.dataset_date = self.dataset
        self.dataset_date.index = pd.to_datetime(self.dataset_date.index, unit='ms')
        self.q_params_df = joblib.load('models/arima_q_paramsDF.pkl')
        self.p_params_df = joblib.load('models/arima_p_paramsDF.pkl')
        self.d_param = 1 #See what will change if d == 0

    def train(self):
        from timeit import default_timer as timer
        start_global_time = timer()
        print("START ARIMA FIT")
        for col_name in self.dataset_date.columns:
            print('==========================================================')
            print('TAG: ', col_name)
            q = self.q_params_df.loc[col_name].q
            p = self.p_params_df.loc[col_name].p
            d = self.d_param
            print('p: ', p, ' d: ', d, ' q: ', q)
            model_ARIMA = ARIMA(self.dataset_date[col_name], order=(p, d, q))

            start_local_time = timer()
            results_ARIMA = model_ARIMA.fit()
            end_local_time = timer()

            #model_AR = ARIMA(self.dataset_date[col_name], order=(p, d, 0))
            #model_MA = ARIMA(self.dataset_date[col_name], order=(0, d, q))
            file_name = 'models/ARIMA/default' + col_name + '_p_is_' + p + '_d_is_' + d + '_q_is_' + q + '.pkl'
            joblib.dump(results_ARIMA, file_name)
            print('Time for fit: ', end_local_time - start_local_time)
            print('==========================================================')
        end_global_time = timer()
        print('FINISHED. TIME IS: ', end_global_time - start_global_time)

if __name__ == '__main__':
    #file = 'data/train_normalized.csv'
    #dataset = utils.parse_train_data(file)
    #qDF = q_params(dataset)
    #save_qparams(qDF)
    #pDF = p_params(dataset)
    #save_pparams(pDF)
    cls = ArimaModel()
    cls.train()


