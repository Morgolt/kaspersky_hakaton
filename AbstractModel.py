class AbstractModel:

    def test(self, dataset):
        """
        Should return timestamp from dataset (first column) where anomaly was detected.
        :param dataset: Pandas DataFrame
        :return: float timestamp
        """
        return dataset.index[0]

    def __repr__(self):
        return self.__class__.__name__
