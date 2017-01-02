from keras import backend as K
from keras.layers import Layer


class AdvantageAggregator(Layer):
    def call(self, inputs, mask=None):
        v = inputs[0]
        a = inputs[1]
        return K.repeat_elements(v, K.shape(a)[1], 1) + a - K.expand_dims(K.mean(a, axis=-1))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]  # "a" shape
