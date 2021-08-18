import tensorflow as tf


class Dice(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],),
                                      initializer=tf.zeros_initializer(), dtype=tf.float32,
                                      name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # tf.layers.batch_normalization(
        # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
