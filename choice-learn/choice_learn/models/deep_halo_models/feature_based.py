"""Feature Deep Halo model implemented in TensorFlow."""

import tensorflow as tf

def make_valid_mask(n, lengths, mode='without'):
    # lengths: (batch,)
    row = tf.range(n)[tf.newaxis, :]  # (1, n)
    lengths = tf.cast(lengths, tf.int32)
    if mode == 'with':
        return tf.logical_or(row < lengths[:, tf.newaxis], row == n - 1)
    else:
        return row < lengths[:, tf.newaxis]

def masked_softmax(scores, lengths):
    n = tf.shape(scores)[1]
    valid = make_valid_mask(n, lengths)
    scores = tf.where(valid, scores, tf.fill(tf.shape(scores), float('-inf')))
    return tf.nn.log_softmax(scores, axis=-1)

class NonlinearTransformation(tf.keras.layers.Layer):
    def __init__(self, H, embed=128, dropout=0.0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(embed * H)
        self.fc2 = tf.keras.layers.Dense(embed)
        self.H = H
        self.embed = embed
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.enc_norm = tf.keras.layers.LayerNormalization()

    def call(self, X, training=False):
        # X: (B, n, embed)
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]
        X = self.fc1(X)  # (B, n, embed*H)
        X = tf.reshape(X, (B, n, self.H, self.embed))  # (B, n, H, embed)
        X = tf.nn.relu(X)
        X = self.dropout(X, training=training)
        X = self.fc2(X)
        X = self.enc_norm(X)
        return X

class FeatureDeepHalo(tf.keras.Model):
    def __init__(self, n, input_dim, H, L, embed=128, dropout=0.0):
        super().__init__()
        self.basic_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embed, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed)
        ])
        self.enc_norm = tf.keras.layers.LayerNormalization()
        self.aggregate_linear = [tf.keras.layers.Dense(H) for _ in range(L)]
        self.nonlinear = [NonlinearTransformation(H, embed, dropout) for _ in range(L)]
        self.H = H
        self.embed = embed
        self.final_linear = tf.keras.layers.Dense(1)
        self.qualinear1 = tf.keras.layers.Dense(embed)
        self.qualinear2 = tf.keras.layers.Dense(embed)
        self.L = L

    def call(self, X, lengths, training=False):
        # X: (B, n, input_dim)
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]
        Z = self.enc_norm(self.basic_encoder(X, training=training))  # (B, n, embed)
        X_ = tf.identity(Z)
        for fc, nt in zip(self.aggregate_linear, self.nonlinear):
            # (B, n, H)
            Z_bar = tf.reduce_sum(fc(Z), axis=1) / tf.cast(lengths[:, tf.newaxis], tf.float32)  # (B, H)
            Z_bar = Z_bar[:, tf.newaxis, :, tf.newaxis]  # (B, 1, H, 1)
            phi = nt(X_, training=training)  # (B, n, H, embed)
            valid = make_valid_mask(n, lengths)  # (B, n)
            valid = tf.cast(valid, tf.float32)
            phi = phi * valid[:, :, tf.newaxis, tf.newaxis]  # (B, n, H, embed)
            Z = tf.reduce_sum(phi * Z_bar, axis=2) / self.H + Z  # (B, n, embed)
        logits = self.final_linear(Z)  # (B, n, 1)
        logits = tf.squeeze(logits, axis=-1)  # (B, n)
        probs = masked_softmax(logits, lengths)  # (B, n)
        return probs, logits