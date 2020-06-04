import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_gen.datagenerator import TCNDataGenerator, TCNClassGenerator, \
    EDTCNGenerator, StepTCNGenerator, ParamTCNGenerator
from utility.save_load_util import document_model, incr_dir
from model_selection.architectures import *
from tcn_predict import plot_pred, plot_history
from data_gen.preproc import norm_emg, bp_filter, spec_proc, smooth, lms_anc


def callback_gen(dir_path, end, patience=8, verbose=(1, 1)):
    mc = ModelCheckpoint(dir_path + '\\best_model_' + str(end) +
                         '.h5', monitor='val_loss', mode='min', verbose=verbose[0], save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose[1], patience=patience)
    return [mc, es]


def main():
    data_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB003\Data'
    val_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB003\Validation'
    target_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\Models'
    generator = TCNDataGenerator
    model_type = depthwise_model

    def target_files(f):
        return "Walk" in f

    child_dir = 'model_'
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
        'stride': 1,
        'freq_factor': 20,
        'file_names': sorted([f for f in os.listdir(data_dir)
                              if f.endswith('.json') and target_files(f)]),
        'channel_mask': None,
        'time_step': 1,
        ###############################################################################
        'preproc': norm_emg,
        # 'ppkwargs': {'high_band': 35, 'low_band': 200, 'sfreq': 2000, 'filt_ord': 1, 'causal': False},
        # 'ppkwargs': {'fbins': 500, 'gamma': 0.01, 'frange': (4, 250), 'fsamp': 2000},
        # 'ppkwargs': {'clear_ecg': True, 'filter_mask': (4, 5),
        #              'order': 4, 'pad': True,
        #              'mu': 0.5, 'eps': 1/0.5, 'rho': 0, 'act': 'tanh', 'bias': 0.1, 'pretrain': (400, 3)},
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
    vvv = tcn_generator.get_k(0)
    val_params = dict(gen_params)
    val_params['data_dir'] = val_dir
    val_params['file_names'] = sorted([f for f in os.listdir(val_dir) if f.endswith('.json') and target_files(f)])
    val_generator = generator(**val_params)
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

        'drp': 0.4,
        'dense_drp': True,
        'dense': (1000, 200, 200, 100, 50),
        'b_norm': True
    }

    model = model_type(**model_params)
    model.summary()

    # old_model_num = 221
    # old_model = load_model('Models/model_' + str(old_model_num) + '/best_model_' + str(old_model_num) + '.h5')
    # for _ in range(18):
    #     old_model.pop()
    # old_model.save_weights('old_weights.h5')
    # model.load_weights('old_weights.h5', by_name=True)
    # for i in range(10):
    #     model.layers[i].trainable = True
    # if model_type is depthwise_model_class:
    #     model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
    # else:
    #     model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['mape'])

    # Train model on dataset
    cb = callback_gen(dir_path=new_dir, end=max(ends) + 1, patience=10)
    history = model.fit(tcn_generator, callbacks=cb, validation_data=val_generator, verbose=1, epochs=200)
    # endregion

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Save and document
    document_model(new_dir, max(ends) + 1, model, history, **{**gen_params, **model_params})
    val_generator.save(new_dir + '\\gen_' + str(max(ends) + 1) + '.pickle', unload=True)

    plot_history(history, new_dir)

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


def cross_validate(k, generator, model_type, model_params):
    return


if __name__ == '__main__':
    main()
