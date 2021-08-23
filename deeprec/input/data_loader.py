
class DataSet(object):
    def __init__(self, dense_feature, sparse_feature):
        self._data = None
        self._dense_feature_names = None
        self._sparse_feature_names = None
        self._dense_feat = dense_feature
        self._sparse_feat = sparse_feature

    @property
    def data(self):
        return self._data

    @property
    def feature_names(self):
        return self._dense_feature_names, self._sparse_feature_names
