import tensorflow as tf
from .base import BaseModel


class DNN(BaseModel):
    def __init__(self, dense_feature, sparse_feature, task=None, hiddens=None):
        super(DNN, self).__init__(dense_feature, sparse_feature, task)
        pass
