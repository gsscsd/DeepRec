import tensorflow as tf
from .base import BaseModel


class FM(BaseModel):
    def __init__(self, dense_feature, sparse_feature, task=None):
        super(FM, self).__init__(dense_feature, sparse_feature, task)
        pass
