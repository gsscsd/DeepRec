class BaseFeature(object):

    def __init__(self, name, dim, dtype, **kwargs):
        self._name = name
        self._dim = dim
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return "BaseFeature name: {0}, dim: {1}, dtype: {2}".format(self._name, self._dim, self._dtype)


class DenseFeature(BaseFeature):
    def __init__(self, name, dim, dtype, func, **kwargs):
        self._func = func
        super(DenseFeature, self).__init__(name, dim, dtype, **kwargs)

    @property
    def func(self):
        return self._func


class SparseFeature(BaseFeature):
    def __init__(self, name, dim, dtype, use_hash, hash_size, emb_dim, **kwargs):
        self._use_hash = use_hash
        self._hash_size = hash_size
        self._emb_dim = emb_dim
        super(SparseFeature, self).__init__(name, dim, dtype, **kwargs)

    @property
    def use_hash(self):
        return self._use_hash

    @property
    def hash_size(self):
        return self._hash_size

    @property
    def emb_dim(self):
        return self._emb_dim
