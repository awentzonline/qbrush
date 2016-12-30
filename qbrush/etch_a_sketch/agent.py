from keras.layers import (
    Activation, Convolution2D, Dense, Dropout, Flatten, Input, LeakyReLU, merge
)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from qbrush.agent import QAgent


class EtchASketchAgent(QAgent):
    def build_model(self):
        # input: position, canvas, target
        # position
        p = position_in = Input(shape=(2,))
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        merged_inputs = merge([canvas_in, target_in], mode='concat')
        x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same')(merged_inputs)
        x = LeakyReLU()(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        # combine
        y = merge([p, x], mode='concat')
        y = Dense(256)(y)
        y = Dropout(0.3)(y)
        y = LeakyReLU()(y)
        y = Dense(self.num_actions)(y)
        self.model = Model([position_in, canvas_in, target_in], y)
        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        print self.model.summary()
