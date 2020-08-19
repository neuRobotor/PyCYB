import numpy as np
import os
import json
import re
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, DepthwiseConv2D, Input, Concatenate
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from functools import partial
from Archive.convnet import summary
from Archive.convmemnet import norm_emg, stack_emg, data_proc
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import normalize


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
def depthwise_model_gap(shape_X1, shape_X2, shape_Y, drp=0.3, krnl=(3, 3), dilate=0, mpool=0):
    n_timesteps1 = shape_X1[2]
    n_timesteps2 =shape_X2[2]
    n_channels = shape_X1[3]
    n_outputs = shape_Y[1]

    input_old = Input((1, n_timesteps1, n_channels))
    input_recent = Input((1, n_timesteps2, n_channels))

    conv_layer1 = DepthwiseConv2D(input_shape=(1, n_timesteps1, n_channels),
                                  kernel_size=(1, krnl[0]),
                                  depth_multiplier=4,
                                  activation='relu',
                                  padding='valid')

    pool = MaxPooling2D(pool_size=(1, mpool))

    if not dilate:
        conv_layer2 = DepthwiseConv2D(input_shape=(1, n_timesteps2, n_channels),
                                      kernel_size=(1, krnl[1]),
                                      depth_multiplier=16,
                                      activation='relu',
                                      padding='valid')
    else:
        conv_layer2 = DepthwiseConv2D(input_shape=(1, n_timesteps2, n_channels),
                                      kernel_size=(1, krnl[1]),
                                      dilation_rate=dilate,
                                      depth_multiplier=4,
                                      activation='relu',
                                      padding='valid')

    featuremap1 = conv_layer2(pool(conv_layer1(input_recent)))
    featuremap2 = conv_layer2(pool(conv_layer1(input_old)))

    if mpool:
        pool = MaxPooling2D(pool_size=(1, mpool))
        featuremap1 = pool(featuremap1)
        featuremap2 = pool(featuremap2)

    drop = Dropout(drp)
    featuremap1 = drop(featuremap1)
    featuremap2 = drop(featuremap2)

    flat1 = Flatten()(featuremap1)
    flat2 = Flatten()(featuremap2)
    features = Concatenate()([flat1, flat2])
    out = Dense(100, activation='relu')(features)
    out = Dense(50, activation='relu')(out)
    out = Dense(n_outputs, activation='linear')(out)
    model = Model([input_old, input_recent], out)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    return model


def train_net(X, Y, dil, drop, poolsize, kernel, ep, ba, k, validate, window_size, stride, freq_factor, windows):
    cur_model = partial(depthwise_model_gap, shape_X1=X[0].shape, shape_X2=X[1].shape, shape_Y=Y.shape, drp=drop, krnl=kernel, dilate=dil,
                        mpool=poolsize)
    #cur_model =  cur_model = partial(depthwise_model, shape_X=X[1].shape, shape_Y=Y.shape, drp=drop, krnl=kernel, dilate=dil, mpool=poolsize)
    model = cur_model()

    if not validate:
        file_path = r'C:\Users\hbkm9\Documents\Projects\CYB\Balint\CYB104\Data\104_Validation40.json'
        with open(file_path) as json_file:
            dict_data = json.load(json_file)
        emg_data = norm_emg(np.array(dict_data["EMG"]))
        X0 = stack_emg(emg_data, window_size=window_size, stride=stride, windows=windows)
        #X0 = X0[:, :, :-1]
        joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
        Y0 = [[] for _ in range(len(joint_names))]
        for i, joint in enumerate(joint_names):
            cur_data = dict_data[joint]
            upsampled = np.array(cur_data)
            xp = np.arange(len(upsampled)) * freq_factor / stride
            x = np.arange(window_size, len(upsampled) * freq_factor + 1)
            x = x[0::stride]

            def interp_f(data):
                return np.interp(x, xp, data)

            upsampled = np.apply_along_axis(interp_f, 0, upsampled)
            Y0[i].extend(upsampled.tolist())

        Y0 = np.array(Y0)
        Y0 = Y0[:, :, 0].transpose()
        Y0 = normalize(np.array(Y0), axis=1)
        # Y0 = np.expand_dims(Y0[:, 2], 1)
        for i in range(2):
            X0[i] = np.expand_dims(X0[i], 1)
        mc = ModelCheckpoint('../Models/best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
        history = model.fit(X, Y, batch_size=ba, epochs=ep, verbose=2, callbacks=[es, mc], validation_data=(X0, Y0))
        ends = [int(re.search(r'(\d+)$', str(os.path.splitext(f)[0])).group(0))
                for f in os.listdir(r'/Models') if f.endswith('.h5')
                and 'model_' in f]
        if not ends:
            ends = [0]
        filepath = r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models\model_' + str(max(ends) + 1) + '.h5'
        model.save(filepath)
        import pickle
        with open(r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models\history_' + str(
                max(ends) + 1) + r'.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    estimator = KerasRegressor(build_fn=cur_model, epochs=ep, batch_size=ba, verbose=2)
    kfold = KFold(n_splits=k)
    scores = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=3)
    return scores, model


def kfold():
    # ########################
    #       LOAD INPUT
    # ########################
    data_path = r'C:\Users\hbkm9\Documents\Projects\CYB\Balint\CYB104\Data'
    window_size = 240
    n_channels = 8
    stride = 1
    freq_factor = 20
    windows = (100, 100)
    diff = "w" in [f for f in os.listdir(data_path) if f.endswith('.json')][0]

    X, Y, files = data_proc(data_path, norm_emg, diff=diff,
                            window_size=window_size, n_channels=n_channels, task='Walk', stride=stride, windows=windows)
    # X = X[:, :, :-1]
    # Y = np.expand_dims(Y[:, 2], 1)
    Y = normalize(np.array(Y), axis=1)
    for i in range(2):
        X[i] = np.expand_dims(X[i], 1)
    print('Data loaded. Beginning training.')

    # ########################
    #       TRAIN NETWORK
    # ########################

    k = 5
    drop = 0.5
    kernel = (15, 3)
    dil = 3
    poolsize = 4
    ep, ba = 25, 20
    validate = False
    if validate:
        scores, model = train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba,
                                  validate=validate, window_size=window_size, stride=stride, freq_factor=freq_factor, windows=windows)
        summary(k, scores, kernel, drop, model, data_path, ep, ba, files)
    else:
        train_net(X, Y, k=k, dil=dil, poolsize=poolsize, kernel=kernel, drop=drop, ep=ep, ba=ba,
                  validate=validate, window_size=window_size, stride=stride, freq_factor=freq_factor, windows=windows)


def main():
    kfold()
    pass


if __name__ == "__main__":
    main()
