import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_gen.datagenerator import TCNDataGenerator, TCNClassGenerator, \
    EDTCNGenerator, StepTCNGenerator, ParamTCNGenerator
from utility.save_load_util import document_model, incr_dir
from model_selection.architectures import *
from tcn_predict import plot_pred, plot_history
from data_gen.preproc import norm_emg, bp_filter, spec_proc, smooth, lms_anc, unwrap_shift
import numpy as np
from model_selection.model_compare import revaluate
import tensorflow as tf


def main(data_dir=None, target_files=None, old_layers=None):
    if data_dir is None:
        data_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data'
    val_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Validation'
    target_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models'
    child_dir = 'model_'
    if target_files is None:
        def target_files(f):
            return "DownSit" in f

    generator = TCNDataGenerator
    model_type = depthwise_model
    cross_val_k = 5
    make_plot = True
    new_dir, ends = incr_dir(target_dir, child_dir, make=True)

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Preprocessing
    gen_params = {
        'data_dir': data_dir,
        ###################
        'window_size': 1000,
        'delay': 1000,
        'gap_windows': None,
        ###################
        'stride': 20,
        'freq_factor': 20,
        'file_names': sorted([f for f in os.listdir(data_dir)
                              if f.endswith('.json') and target_files(f)]),
        'channel_mask': None,
        'time_step': 1,
        ###############################################################################
        'preproc': norm_emg,
        # 'ppkwargs': {'high_band': 35, 'low_band': 200, 'sfreq': 2000, 'filt_ord': 4, 'causal': True},
        # 'ppkwargs': {'fbins': 500, 'gamma': 0.01, 'frange': (4, 250), 'fsamp': 2000},
        # 'ppkwargs': {'clear_ecg': True, 'filter_mask': None,
        #              'order': 4, 'pad': True,
        #              'mu': 0.5, 'eps': 1/0.5, 'rho': 0, 'act': 'tanh', 'pretrain': (400, 3)},
        ###############################################################################
        'batch_size': 64,
        ###############################################################################
        # Classifier params
        # 'class_enum': ('Walk', 'UpSit', 'UpStair', 'DownSit', 'DownStair')
        ###############################################################################
        # EDTCN params
        # 'decode_length': 400
        ###############################################################################
        # Param regression
        # 'params': ('stride_lengths',)
    }
    tcn_generator = generator(**gen_params)
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
        'dil': ((1, 1), (1, 10)),
        'strides': ((1, 1), (1, 1)),
        'mpool': ((1, 6), (1, 5)),  # (feature, time)
        'depth_mul': (5, 3),

        'drp': 0.0,
        'dense_drp': True,
        'dense': (200, 5),
        'b_norm': False
    }

    model = model_type(**model_params)
    model.summary()

    if cross_val_k is not None:
        model, history, metrics = cross_validate(cross_val_k, tcn_generator, model_type, model_params, patience=25,
                                                 selection_methods=(np.min,), model_save_dir=new_dir, old_model=old_layers)
        tcn_generator.save(new_dir + '\\gen_' + str(max(ends) + 1) + '.pickle', unload=True)
        document_model(new_dir, max(ends) + 1, model, history,
                       **{**gen_params, **model_params, **metrics}, datagenerator=generator)
        plot_history(history[0], new_dir)
        return
    else:
        val_params = dict(gen_params)
        val_params['data_dir'] = val_dir
        val_params['file_names'] = sorted([f for f in os.listdir(val_dir) if f.endswith('.json') and target_files(f)])
        val_generator = generator(**val_params)

        if old_layers is not None:
            model = learning_transfer(model, old_model_num=old_layers, n_old_layers=10, trainable=True,
                                      classify=model_type is depthwise_model_class)

        # Train model on dataset
        cb = callback_gen(dir_path=new_dir, end=max(ends) + 1, patience=10)
        history = model.fit(tcn_generator, callbacks=cb, validation_data=val_generator, verbose=1, epochs=3)
    # endregion

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Save and document
    val_generator.save(new_dir + '\\gen_' + str(max(ends) + 1) + '.pickle', unload=True)
    document_model(new_dir, max(ends) + 1, model, history.history, **{**gen_params, **model_params},
                   datagenerator=generator)
    plot_history(history.history, new_dir)

    if not make_plot:
        return
    if val_generator.stride is not 1:
        val_generator.stride = 1
        val_generator.load_files()
    emg_data, Y0 = val_generator.data_generation(range(val_generator.window_index_heads[-1][-1]))
    if len(tcn_generator.emg_data[0]) < 8:
        plot_pred(emg_data=emg_data, Y0=Y0, delay=val_generator.delay, stride=val_generator.stride,
                  ecg=val_generator.emg_data[0][0], data_dir=new_dir, model=model)
    else:
        plot_pred(emg_data=emg_data, Y0=Y0, delay=val_generator.delay, stride=val_generator.stride,
                  ecg=val_generator.emg_data[0][7], data_dir=new_dir, model=model)

    # endregion
    return


def cross_validate(k, generator: TCNDataGenerator, model_type, model_params, patience,
                   selection_methods=(np.min,), old_model=None, n_old_layers=7, model_save_dir=None):
    metrics = {}
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import History
    model = Sequential()
    history = History()
    histories = list()
    mse = list()
    rsq = list()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    if model_type is not depthwise_model_class:
        generator.force_unwrap()
    for cur_k in range(k):
        valid_gen = generator.get_k(cur_k=cur_k, k=k, file_shuffle=True)
        model = model_type(**model_params)
        for i in range(9):
            model.layers[i].trainable=True
        if old_model is not None:
            model = cross_val_transfer(model, old_model_num=old_model, k_num=cur_k, n_old_layers=n_old_layers,
                                       trainable=False, classify=model_type is depthwise_model_class)
        history = model.fit(generator, callbacks=[es, mc], validation_data=valid_gen, verbose=1, epochs=200)
        model = load_model('best_model.h5')
        if model_save_dir is not None:
            model.save(model_save_dir+'\\model_'+ str(cur_k) + '.h5')
        cur_mse, cur_rsq = revaluate(valid_gen, model)
        mse.append(cur_mse)
        rsq.append(cur_rsq)
        if len(selection_methods) == 1:
            selection_methods = selection_methods * len(history.history)

        if len(history.history) % len(selection_methods) == 0:
            selection_methods = selection_methods * int(len(history.history) / len(selection_methods))

        for i, (key, item) in enumerate(history.history.items()):
            if key not in metrics:
                metrics[key] = list()
            metrics[key].append(selection_methods[i](item))
        histories.append(history.history)

    for key in history.history.keys():
        metrics[key] = (metrics[key], np.mean(metrics[key]))
    metrics['mse'] = (np.mean(np.vstack(mse), axis=0), np.std(np.vstack(mse), axis=0))
    metrics['rsq'] = (np.mean(np.vstack(rsq), axis=0), np.std(np.vstack(rsq), axis=0))
    metrics['mse_stack'] = np.vstack(mse)
    metrics['rsq_stack'] = np.vstack(rsq)

    return model, histories, metrics


def cross_val_transfer(model, old_model_num, k_num, n_old_layers=10, trainable=True, classify=False):
    old_model = load_model('Models/model_' + str(old_model_num) + '/model_' + str(k_num) + '.h5')
    for _ in range(len(old_model.layers) - n_old_layers):
        old_model.pop()
    for i in range(n_old_layers):
        old_model.layers[i]._name = model.layers[i]._name
    old_model.save_weights('old_weights.h5')
    model.load_weights('old_weights.h5', by_name=True)
    for i in range(n_old_layers):
        model.layers[i].trainable = trainable
    if classify:
        model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
    else:
        model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mape'])
    return model


def learning_transfer(model, old_model_num, n_old_layers=10, trainable=True, classify=False):
    old_model = load_model('Models/model_' + str(old_model_num) + '/best_model_' + str(old_model_num) + '.h5')
    for _ in range(len(old_model.layers) - n_old_layers):
        old_model.pop()
    old_model.save_weights('old_weights.h5')
    model.load_weights('old_weights.h5', by_name=True)
    for i in range(n_old_layers):
        model.layers[i].trainable = trainable
    if classify:
        model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
    else:
        model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mape'])
    return model


def callback_gen(dir_path, end, patience=8, verbose=(1, 1)):
    mc = ModelCheckpoint(dir_path + '\\best_model_' + str(end) +
                         '.h5', monitor='val_loss', mode='min', verbose=verbose[0], save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose[1], patience=patience)
    return [mc, es]


if __name__ == '__main__':
    data_dirs = ( r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data',)
    file_names = ("Walk",)
    # old_models = (473, 474, 475, 476)
    for data_dir in data_dirs:
        for file_name in file_names:
            def target(f):
                return file_name in f

            main(data_dir=data_dir, target_files=target)
