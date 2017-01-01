from keras import backend as K
from keras.layers import Layer


class AdvantageAggregator(Layer):
    def call(self, x, mask=None):
        return x - K.expand_dims(K.mean(x, axis=-1))
