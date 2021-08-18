# -*- coding:utf-8 -*-
"""
"""

import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, eps=1e-9, center=True,
                 scale=True, **kwargs):
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scale = scale
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.sqrt(variance + self.eps)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'eps': self.eps, 'center': self.center, 'scale': self.scale}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))