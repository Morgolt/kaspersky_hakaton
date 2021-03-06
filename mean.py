import AbstractModel
import utils
import numpy as np
import normalization
import pandas

class MeansModel(AbstractModel.AbstractModel):

    def test(self, dataset):
        # normalization.z_normalize(dataset)
        subed = np.abs(dataset.sub(self.means))
        sumed = subed.sum(axis=1)
        result = sumed.idxmax() if not sumed.empty else -1
        print("Tested")
        return result

    def train(self):
        data = utils.parse_train_data('data/train.csv')
        normalization.make_correlation_free_set(data)
        # normalization.z_normalize(data)
        self.means = data.mean()
        print("Trained")


if __name__ == "__main__":
    model = MeansModel()
    model.train()
    utils.test_model(model)
