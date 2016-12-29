from keras.layers import (
    Activation, Convolution2D, Dense, Flatten, Input, LeakyReLU, merge
)
from keras.models import Model, Sequential
from qbrush.agent import QAgent


class EtchASketchAgent(QAgent):
    def build_model(self):
        # input: position, canvas, target
        # position
        p = position_in = Input(shape=(2,))
        # canvas and target
        conv = Sequential()
        conv.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same', input_shape=self.canvas_shape))
        conv.add(LeakyReLU())
        conv.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same'))
        conv.add(LeakyReLU())
        conv.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same'))
        conv.add(LeakyReLU())
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        c = conv(canvas_in)
        t = conv(target_in)
        c = Flatten()(c)
        t = Flatten()(t)
        # combine
        y = merge([p, c, t], mode='concat')
        y = Dense(256)(y)
        y = LeakyReLU()(y)
        y = Dense(self.num_actions)(y)
        self.model = Model([position_in, canvas_in, target_in], y)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        print self.model.summary()
