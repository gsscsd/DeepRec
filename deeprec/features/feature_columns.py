import tensorflow as tf


class BaseFeature(object):
    def __init__(self, name, dim=1, dtype=tf.float32, **kwargs):
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
    def __init__(self, name, dim=1, dtype=tf.float32, func=None, **kwargs):
        self._func = func
        super(DenseFeature, self).__init__(name, dim, dtype, **kwargs)

    @property
    def func(self):
        return self._func

    def __str__(self):
        return "DenseFeature name: {0}, dim: {1}, dtype: {2}".format(self.name, self.dim, self.dtype)


class SparseFeature(BaseFeature):
    def __init__(self, name, dim=1, dtype=tf.float32, vocab_size=None, hash_size=None, emb_size=None, **kwargs):
        self._hash_size = hash_size
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        super(SparseFeature, self).__init__(name, dim, dtype, **kwargs)

    @property
    def hash_size(self):
        return self._hash_size

    @property
    def vocab_size(self):
        return self._vocab_size
    @property
    def emb_size(self):
        return self._emb_size

    def __str__(self):
        return "SparseFeature name: {0}, dim: {1}, dtype: {2}, " \
               "hash_size: {3}, vocab_size : {4}, emb_size: {5}".format(self.name,
                                                                        self.dim,
                                                                        self.dtype,
                                                                        self.hash_size,
                                                                        self.vocab_size,
                                                                        self.emb_size)