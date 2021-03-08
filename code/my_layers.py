import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable(package='Custom', name='ortho')
class OthoRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, x):
        w_n = tf.math.l2_normalize(x)
        reg = tf.reduce_sum(tf.square(tf.matmul(w_n, w_n, transpose_b=True) - tf.eye(w_n.shape[0])))
        return self.rate * reg


class Average(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Average, self).__init__(**kwargs)
        self.supports_masking = True

    def get_config(self):
        base_config = super(Average, self).get_config()
        return base_config

    def call(self, x, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            x = x * mask
        return tf.reduce_sum(x, axis=-2) / tf.reduce_sum(mask, axis=-2)

    def compute_mask(self, x, mask=None):
        return None


class AttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionEncoder, self).__init__(**kwargs)
        self.supports_masking = True

    def get_config(self):
        base_config = super(AttentionEncoder, self).get_config()
        return base_config

    def build(self, input_shape):
        assert len(input_shape) == 2
        hidden_size = input_shape[0][-1]
        self.dense_layer = tf.keras.layers.Dense(
            units=hidden_size,
            name="dense_layer",
        )
        self.mask_softmax = TemporalSoftmax()

    def call(self, inputs, mask=None):
        word_embedding = inputs[0]
        sen_embedding = inputs[1]
        sen_embedding = self.dense_layer(sen_embedding)
        sen_embedding = tf.expand_dims(sen_embedding, 1)
        attention = tf.matmul(sen_embedding, word_embedding, transpose_b=True)
        attention = self.mask_softmax(attention, mask=mask[0])
        output = tf.matmul(attention, word_embedding)
        output = tf.squeeze(output, axis=1)
        return output


class TemporalSoftmax(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -2)
        inputs_exp = tf.exp(inputs) * broadcast_float_mask
        inputs_sum = tf.reduce_sum(inputs_exp, axis=-1, keepdims=True)
        return inputs_exp / inputs_sum


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 aspect_num,
                 hidden_size,
                 dropout_rate=0.0,
                 reg_rate=0.1,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.aspect_num = aspect_num
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.reg_rate = reg_rate
        self.supports_masking = True

    def get_config(self):
        config = {
            "aspect_num": self.aspect_num,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "reg_rate": self.reg_rate,
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.sen_to_aspect_layer = tf.keras.layers.Dense(
            units=self.aspect_num,
            activation="softmax",
            name="sen_to_aspect_layer",
        )
        self.aspect_dropout_layer = tf.keras.layers.Dropout(
            rate=self.dropout_rate,
            name="aspect_dropout_layer",
        )
        self.aspect_to_sen_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_regularizer=OthoRegularizer(self.reg_rate),
            name="aspect_to_sen_layer",
        )

    def call(self, inputs, training=None):
        sen_to_aspect = self.sen_to_aspect_layer(inputs)
        sen_to_aspect = self.aspect_dropout_layer(sen_to_aspect, training)
        aspect_to_sen = self.aspect_to_sen_layer(sen_to_aspect)
        return aspect_to_sen


class MaxMargin(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(MaxMargin, self).get_config()
        return base_config

    def call(self, inputs, mask=None):
        att_emb, rec_emb, neg_emb = inputs

        att_emb = tf.math.l2_normalize(att_emb, axis=-1)
        rec_emb = tf.math.l2_normalize(rec_emb, axis=-1)
        neg_emb = tf.math.l2_normalize(neg_emb, axis=-1)

        pos = tf.math.reduce_sum(tf.math.multiply(rec_emb, att_emb), axis=-1, keepdims=True)
        neg = tf.math.reduce_sum(tf.math.multiply(neg_emb, tf.expand_dims(att_emb, axis=1)), axis=-1)
        loss = tf.reduce_sum(tf.math.maximum(0., (1.0 - pos + neg)), axis=-1, keepdims=True)

        return loss
