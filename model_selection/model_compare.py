from keras.models import Sequential, Model
from keras.layers import DepthwiseConv2D, Reshape, Flatten, Dense, Input, Conv1D
import keras
import numpy as np


def depthwise_output():
    inputs = Input(shape=(1, 20, 8))
    inter = DepthwiseConv2D(input_shape=(1, 20, 8),
                            kernel_size=(1, 4),  # height 1,  width 7  (ostensibly 1D)
                            depth_multiplier=2,
                            depthwise_initializer='ones',
                            bias_initializer='zeros',
                            padding='valid')(inputs)
    inter2 = DepthwiseConv2D(input_shape=(1, 20, 8),
                              kernel_size=(1, 4),  # height 1,  width 7  (ostensibly 1D)
                              dilation_rate=4,
                              depth_multiplier=2,
                              depthwise_initializer='ones',
                              bias_initializer='zeros',
                              padding='valid')(inter)
    outputs = Flatten()(inter2)
    model = Model(inputs=inputs, outputs=[outputs, inter, inter2])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def depthwise_model():
    model = Sequential()
    model.add(DepthwiseConv2D(input_shape=(1, 20, 8),
                              kernel_size=(1, 3),  # height 1,  width 7  (ostensibly 1D)
                              depth_multiplier=16,
                              activation='elu',
                              padding='valid'))
    model.add(Flatten())
    model.add(Dense(6, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def api_model():
    inputs_in = list()
    flattens = list()
    for i in range(8):
        inp = Input(shape=(20, 1))
        inputs_in.append(inp)
        x = Conv1D(filters=16, kernel_size=3, activation='elu', input_shape=(20, 1))(inp)
        x = Flatten()(x)
        flattens.append(x)
    x = keras.layers.concatenate(flattens)
    output = Dense(6, activation='linear')(x)
    model = Model(inputs=inputs_in, outputs=[output])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def convo_model():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(20, 8)))
    model.add(Flatten())
    model.add(Dense(6, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def main():
    m1 = depthwise_model()
    m2 = api_model()
    m3 = convo_model()
    print(m1.summary())
    print(m2.summary())
    print(m3.summary())
    return


if __name__ == "__main__":
    main()
