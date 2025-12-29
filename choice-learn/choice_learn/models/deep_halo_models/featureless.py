"""Featureless Deep Halo model implemented in TensorFlow."""

import tensorflow as tf

class ExaResBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_main = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.linear_act = tf.keras.layers.Dense(hidden_dim, use_bias=False, input_shape=(input_dim,))

    def call(self, z_prev, z0):
        return self.linear_main(z_prev * self.linear_act(z0)) + z_prev

class QuaResBlock(tf.keras.layers.Layer):
    def __init__(self, d):
        super().__init__()
        self.linear = tf.keras.layers.Dense(d, use_bias=False)

    def call(self, x):
        return self.linear(tf.math.pow(x, 2)) + x

class FeaturelessDeepHalo(tf.keras.Model):
    def __init__(self, opt_size, depth, resnet_width, block_types):
        super().__init__()
        assert len(block_types) == depth - 1
        self.in_lin = tf.keras.layers.Dense(resnet_width, use_bias=False, input_shape=(opt_size,))
        self.out_lin = tf.keras.layers.Dense(opt_size, use_bias=False)
        self.blocks = []
        for t in block_types:
            if t == "exa":
                self.blocks.append(ExaResBlock(opt_size, resnet_width))
            elif t == "qua":
                self.blocks.append(QuaResBlock(resnet_width))
            else:
                raise ValueError(f"Unknown block type: {t}")

    def call(self, e):
        mask = tf.equal(e, 1)
        e0 = tf.identity(e)
        e = self.in_lin(e)
        for b in self.blocks:
            if isinstance(b, ExaResBlock):
                e = b(e, e0)
            else:
                e = b(e)
        logits = self.out_lin(e)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), float("-inf")))
        probas = tf.nn.softmax(logits, axis=-1)
        return probas, logits
