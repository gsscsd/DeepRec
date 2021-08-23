import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding


class FinalLinear(tf.keras.layers.Layer):
    def __init__(self, activation=None):
        super(FinalLinear, self).__init__()
        self.dense = Dense(1, activation=activation)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result


class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.0):
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class BaseModel(object):
    def __init__(self, dense_feature, sparse_feature, task=None):
        self._dense_inp = {
            feat.name: Input(shape=(feat.dim,), name=feat.name, dtype=feat.dtype)
            for feat in dense_feature
        }
        self._sparse_inp = {
            feat.name: Input(shape=(feat.dim,), name=feat.name, dtype=feat.dtype)
            for feat in sparse_feature
        }
        self.embed_layers = {
            feat.name: Embedding(input_dim=feat.vocab_size,
                                 input_length=1,
                                 output_dim=feat.emb_size,
                                 embeddings_initializer='random_uniform')
            for feat in sparse_feature
        }
        self._model = None
        self._inps = None
        self._task = None
        self._build()


    def _build(self):
        dense_inp = tf.concat(list(self._dense_inp.values()), axis=-1)
        sparse_inp = tf.concat([self.embed_layers[feat](inp) for feat, inp in self._sparse_inp.items()], axis=-1)
        sparse_inp = tf.squeeze(sparse_inp, axis=1)
        inp = tf.concat([dense_inp, sparse_inp], axis=-1)
        out = Dense(units=1, activation='sigmoid')(inp)
        self._model = tf.keras.models.Model(inputs=list(self._dense_inp.values()) + list(self._sparse_inp.values()), outputs=out)

    def get_model(self):
        return self._model
