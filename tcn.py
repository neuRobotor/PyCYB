import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, DepthwiseConv2D, Reshape, Concatenate, \
    Input, BatchNormalization, Activation, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_gen.datagenerator import TCNDataGenerator
import numpy as np
from scipy import signal
from utility.save_load_util import document_model, incr_dir
from tcn_predict import plot_pred, plot_history
from data_gen.preproc import norm_emg, bp_filter, spec_proc


def conv_model(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1 ,3), (1,3)), dil=1,
                    mpool=((0,0), (0,0)), dense=(100, 50), acts=('relu', 'relu'), b_norm=False, dense_drp=False,
                    pad='valid'):
    model = Sequential(name='Conv2D_Model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
        model.add(Conv2D(kernel_size=krnl[0],
                          filters=depth_mul[0],
                          activation=acts[0],
                          padding=pad))
    else:
        model.add(Conv2D(kernel_size=krnl[0],
                          filters=depth_mul[0],
                          activation=acts[0],
                          padding=pad,
                          input_shape=input_shape))
    if mpool[0][0]:
        model.add(MaxPooling2D(pool_size=mpool[0]))

    model.add(Conv2D(kernel_size=krnl[1],
                        filters=depth_mul[1],
                        activation=acts[0],
                        padding=pad,
                        dilation_rate=dil))
    if mpool[1][0]:
        model.add(MaxPooling2D(pool_size=mpool[1]))
    model.add(Dropout(drp))
    model.add(Flatten())
    if b_norm:
        for d in dense:
            model.add(Dense(d))
            model.add(BatchNormalization())
            model.add(Activation(acts[1]))
            if dense_drp:
                model.add(Dropout(drp))
    else:
        for d in dense:
            model.add(Dense(d, activation=acts[1]))
            if dense_drp:
                model.add(Dropout(drp))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mape'])
    return model


def depthwise_model(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1 ,3), (1,3)), dil=1,
                    mpool=((0,0), (0,0)), dense=(100, 50), acts=('relu', 'relu'), b_norm=False, dense_drp=False,
                    pad='valid'):
    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
        model.add(DepthwiseConv2D(kernel_size=krnl[0],
                                  depth_multiplier=depth_mul[0],
                                  activation=acts[0],
                                  padding=pad))
    else:
        model.add(DepthwiseConv2D(kernel_size=krnl[0],
                                  depth_multiplier=depth_mul[0],
                                  activation=acts[0],
                                  padding=pad,
                                  input_shape=input_shape))
    if mpool[0][0]:
        model.add(MaxPooling2D(pool_size=mpool[0]))

    model.add(DepthwiseConv2D(kernel_size=krnl[1],
                              depth_multiplier=depth_mul[1],
                              activation=acts[0],
                              padding=pad,
                              dilation_rate=dil))
    if mpool[1][0]:
        model.add(MaxPooling2D(pool_size=mpool[1]))
    model.add(Dropout(drp))
    model.add(Flatten())
    if b_norm:
        for d in dense:
            model.add(Dense(d))
            model.add(BatchNormalization())
            model.add(Activation(acts[1]))
            if dense_drp:
                model.add(Dropout(drp))
    else:
        for d in dense:
            model.add(Dense(d, activation=acts[1]))
            if dense_drp:
                model.add(Dropout(drp))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mape'])
    return model


def depthwise_model_gap(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1,3), (1,3)), dil=1,
                        mpool=((0, 0), (0, 0)), dense=(100, 50),  acts=('relu', 'relu'), b_norm=False, dense_drp=False,
                        pad='valid'):
    input_shape1, input_shape2 = input_shape

    input_old = Input(input_shape1)
    input_recent = Input(input_shape2)

    if len(input_shape1) < 3:
        input_old = Reshape((1, *input_shape1), input_shape=input_shape1)(input_old)
        input_recent = Reshape((1, *input_shape2), input_shape=input_shape2)(input_recent)

    conv_layer1 = DepthwiseConv2D(kernel_size=(1, krnl[0]),
                                  depth_multiplier=depth_mul[0],
                                  activation=acts[0],
                                  padding=pad)

    pool = MaxPooling2D(pool_size=(1, mpool[0]))

    conv_layer2 = DepthwiseConv2D(kernel_size=(1, krnl[1]),
                                  depth_multiplier=depth_mul[1],
                                  activation=acts[0],
                                  dilation_rate=dil,
                                  padding=pad)

    featuremap1 = conv_layer2(pool(conv_layer1(input_recent)))
    featuremap2 = conv_layer2(pool(conv_layer1(input_old)))

    if mpool:
        pool = MaxPooling2D(pool_size=(1, mpool[1]))
        featuremap1 = pool(featuremap1)
        featuremap2 = pool(featuremap2)

    drop = Dropout(drp)
    featuremap1 = drop(featuremap1)
    featuremap2 = drop(featuremap2)

    flat1 = Flatten()(featuremap1)
    flat2 = Flatten()(featuremap2)
    features = Concatenate()([flat1, flat2])
    out = Dense(dense[0], activation=acts[1])(features)
    if b_norm:
        for d in dense[1:]:
            out = Activation(acts[1])(BatchNormalization()(Dense(d)(out)))
            if dense_drp:
                out = Dropout(drp)(out)
    else:
        for d in dense[1:]:
            out = Dense(d, activation=acts[1])(out)
            if dense_drp:
                out = Dropout(drp)(out)
    out = Dense(n_outputs, activation='linear')(out)
    model = Model([input_old, input_recent], out, name='Depthwise_Gap_Model')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mape'])
    return model


def callback_gen(dir_path, end, patience=8, verbose=(1, 1)):
    mc = ModelCheckpoint(dir_path + '\\best_model_' + str(end) +
                         '.h5', monitor='val_loss', mode='min', verbose=verbose[0], save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose[1], patience=patience)
    return [mc, es]


def main():
    data_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data'
    target_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models'
    child_dir = 'model_'
    new_dir, ends = incr_dir(target_dir, child_dir, make=True)

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Preprocessing
    gen_params = {
        'data_dir': data_dir,
        ###################
        'window_size': 400,
        'delay': 400,
        'gap_windows': None,
        ###################
        'stride': 1,
        'freq_factor': 20,
        'file_names': sorted([f for f in os.listdir(data_dir)
                              if f.endswith('.json') and 'Walk' in f]),
        'channel_mask': None,
        ###############################################################################
        'preproc': norm_emg,
        # 'ppkwargs': {'high_band': 35, 'low_band': 200, 'sfreq': 2000, 'filt_ord': 2, 'causal': True},
        # 'ppkwargs': {'fbins': 1000, 'gamma': 0.01, 'frange': (2, 250), 'fsamp': 2000},
        ###############################################################################
        'batch_size': 64
    }
    tcn_generator = TCNDataGenerator(**gen_params)

    val_params = dict(gen_params)
    val_params['file_names'] = sorted([f for f in os.listdir(data_dir) if f.endswith('.json') and 'Validation' in f])
    val_generator = TCNDataGenerator(**val_params)
    # endregion

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Model building and training
    model_params = {
        'input_shape': tcn_generator.data_generation([0])[0][0].shape if gen_params['gap_windows'] is None else
        [temp[0].shape for temp in tcn_generator.data_generation([0])[0]],
        'n_outputs': tcn_generator.data_generation([0])[1][0].shape[0],
        'acts': ('relu', 'selu'),
        'krnl': ((1, 15), (1, 3)),  # (feature, time)
        'pad': 'same',
        'dil': 10,
        'mpool': ((1, 6), (1, 5)),  # (feature, time)
        'depth_mul': (5, 3),

        'drp': 0.4,
        'dense_drp': True,
        'dense': (200, 200, 100, 50),
        'b_norm': True
    }

    model = depthwise_model(**model_params)

    # Train model on dataset
    cb = callback_gen(dir_path=new_dir, end=max(ends)+1, patience=10)
    history = model.fit(tcn_generator, callbacks=cb, validation_data=val_generator, verbose=2, epochs=60)
    # endregion

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Save and document
    document_model(new_dir, max(ends)+1, model, history, **{**gen_params, **model_params})
    tcn_generator.save(new_dir + '\\gen_' + str(max(ends)+1) + '.pickle', unload=True)

    emg_data, Y0 = val_generator.data_generation(range(val_generator.window_index_heads[-1][-1]))
    plot_pred(emg_data=emg_data, Y0=Y0, delay=val_generator.delay, ecg=val_generator.emg_data[0][7], data_dir=new_dir,
              model=model)
    plot_history(history, new_dir)
    # endregion
    return


if __name__ == '__main__':
    main()
