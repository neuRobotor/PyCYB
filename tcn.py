import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_gen.datagenerator import TCNDataGenerator, TCNClassGenerator, EDTCNGenerator
from utility.save_load_util import document_model, incr_dir
from model_selection.architectures import *
from tcn_predict import plot_pred, plot_history
from data_gen.preproc import norm_emg, bp_filter, spec_proc


def callback_gen(dir_path, end, patience=8, verbose=(1, 1)):
    mc = ModelCheckpoint(dir_path + '\\best_model_' + str(end) +
                         '.h5', monitor='val_loss', mode='min', verbose=verbose[0], save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose[1], patience=patience)
    return [mc, es]


def main():
    data_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Data'
    val_dir = r'C:\Users\hbkm9\Documents\Projects\CYB\Experiment1\CYB004\Validation'
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
        'stride': 20,
        'freq_factor': 20,
        'file_names': sorted([f for f in os.listdir(data_dir)
                              if f.endswith('.json') and 'Walk' in f]),
        'channel_mask': None,
        'time_step': 1,
        ###############################################################################
        'preproc': norm_emg,
        # 'ppkwargs': {'high_band': 35, 'low_band': 200, 'sfreq': 2000, 'filt_ord': 1, 'causal': False},
        # 'ppkwargs': {'fbins': 500, 'gamma': 0.01, 'frange': (4, 250), 'fsamp': 2000},
        ###############################################################################
        'batch_size': 64,
        ###############################################################################
        # Classifier params
        # 'class_enum': ('Walk', 'UpSit', 'UpStair', 'DownSit', 'DownStair')
        ###############################################################################
        # EDTCN params
        # 'decode_length': 400
    }
    tcn_generator = TCNDataGenerator(**gen_params)

    val_params = dict(gen_params)
    val_params['data_dir'] = val_dir
    val_params['file_names'] = sorted([f for f in os.listdir(val_dir) if f.endswith('.json') and 'Walk' in f])
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
        'dil': ((1, 1), (1, 10)),
        'strides': ((1, 1), (1, 1)),
        'mpool': ((1, 6), (1, 5)),  # (feature, time)
        'depth_mul': (5, 3),

        'drp': 0.4,
        'dense_drp': True,
        'dense': (500, 200, 200, 100, 50),
        'b_norm': True
    }

    model = depthwise_model(**model_params)
    model.summary()
    # Train model on dataset
    cb = callback_gen(dir_path=new_dir, end=max(ends) + 1, patience=30)
    history = model.fit(tcn_generator, callbacks=cb, validation_data=val_generator, verbose=1, epochs=150)
    # endregion

    # ---------------------------------------------------------------------------------------------------------------- #

    # region Save and document
    document_model(new_dir, max(ends) + 1, model, history, **{**gen_params, **model_params})
    val_generator.save(new_dir + '\\gen_' + str(max(ends) + 1) + '.pickle', unload=True)

    if val_generator.stride is not 1:
        val_generator.stride = 1
        val_generator.load_files()
    emg_data, Y0 = val_generator.data_generation(range(val_generator.window_index_heads[-1][-1]))

    plot_pred(emg_data=emg_data, Y0=Y0, delay=val_generator.delay, stride=val_generator.stride,
              ecg=val_generator.emg_data[0][7], data_dir=new_dir, model=model)
    plot_history(history, new_dir)
    # endregion
    return


if __name__ == '__main__':
    main()
