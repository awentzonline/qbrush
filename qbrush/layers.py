from keras import backend as K
from keras.layers import Layer


class BroadcastDim(Layer):
    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BroadcastDim, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.repeat_elements(x, self.output_size, -1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.output_size,)


class ImageNetMean(Layer):
    def call(self, x, mask=None):
        return x - 120.
