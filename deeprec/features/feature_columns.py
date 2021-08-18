
class BaseFeature(object):

    def __init__(self, name, dim, dtype, **kwargs):
        self._name = name
        self._dim = dim
        self._dtype = dtype

    def get_name(self):
        pass

    def get_dim(self):
        pass

    def get_dtype(self):
        pass

    def __str__(self):
        return "BaseFeature name: {0}, dim: {1}, dtype: {2}".format(self._name, self._dim, self._dtype)


class DenseFeature(BaseFeature):
    def __init__(self):
        pass
    pass


class SparseFeature(BaseFeature):
    def __init__(self):
        pass
    pass