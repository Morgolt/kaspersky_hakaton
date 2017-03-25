import AbstractModel
import utils
import numpy as np
import math

class MyAbstractModel(AbstractModel.AbstractModel):

    def test(self, dataset):


if __name__ == "__main__":
    data = utils.parse_train_data('data/train_reduced.csv')
    means = {key: val.mean() for key, val in data.iteritems() if key != 'Time'}
    for key, val in means.i
    col = np.array(data['tag00'])
    col -= means['tag00']
    col = math.fabs(col)
    print(max(col))
