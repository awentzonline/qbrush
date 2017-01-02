from keras.layers import Layer


class ImageNetMean(Layer):
    def call(self, x, mask=None):
        return x - 120.
