from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, DepthwiseConv2D, Reshape, Concatenate, \
    Input, BatchNormalization, Activation, Conv2D, InputLayer, UpSampling2D, AveragePooling2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K


def conv_model(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1, 3), (1, 3)), dil=((1, 1), (1, 1)),
               mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu'), b_norm=False, dense_drp=False,
               pad='valid', strides=((1, 1), (1, 1))):
    model = Sequential(name='Conv2D_Model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
        model.add(Conv2D(kernel_size=krnl[0],
                                  filters=depth_mul[0],
                                  activation=acts[0],
                                  padding=pad,
                                  dilation_rate=dil[0],
                                  strides=strides[0]))
    else:
        model.add(Conv2D(kernel_size=krnl[0],
                                  activation=acts[0],
                                  padding=pad,
                                  input_shape=input_shape,
                                  dilation_rate=dil[0],
                                  strides=strides[0],
                         filters=depth_mul[0]))
    if mpool[0][0]:
        model.add(MaxPooling2D(pool_size=mpool[0]))

    model.add(Conv2D(kernel_size=krnl[1],
                     filters=depth_mul[1],
                     activation=acts[0],
                     padding=pad,
                     dilation_rate=dil[1],
                     strides=strides[1]))
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
    model.compile(loss=MeanSquaredError(), optimizer=Adam())
    return model


def depthwise_model(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1, 3), (1, 3)), dil=((1, 1), (1, 1)),
                    mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu'), b_norm=False, dense_drp=False,
                    pad='valid', strides=((1, 1), (1, 1))):
    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer( input_shape=input_shape))

    for i in range(len(krnl)):
        model.add(DepthwiseConv2D(kernel_size=krnl[i],
                                  depth_multiplier=depth_mul[i],
                                  activation=acts[0],
                                  padding=pad,
                                  strides=strides[i],
                                  dilation_rate=dil[i]))
        if mpool[i][0]:
            model.add(MaxPooling2D(pool_size=mpool[i]))

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
    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_ed_model(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1, 3), (1, 3)), dil=((1, 1), (1, 1)),
                    mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu'), b_norm=False, dense_drp=False,
                    pad='valid', strides=((1, 1), (1, 1))):
    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

    for i in range(len(krnl)):
        model.add(DepthwiseConv2D(kernel_size=krnl[i],
                                  depth_multiplier=depth_mul[i],
                                  activation=acts[0],
                                  padding=pad,
                                  strides=strides[i],
                                  dilation_rate=dil[i]))
        if mpool[i][0]:
            model.add(MaxPooling2D(pool_size=mpool[i]))

    for i in range(len(krnl)):
        if mpool[i][0]:
            model.add(UpSampling2D(size=mpool[len(krnl)-1-i]))
        model.add(Conv2D(kernel_size=mpool[len(krnl)-1-i],
                                  filters=n_outputs,
                                  activation=acts[1],
                                  padding=pad,
                                  strides=strides[len(krnl)-1-i],
                                  dilation_rate=dil[len(krnl)-1-i]))
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


def depthwise_model_class(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1, 3), (1, 3)),
                          dil=((1, 1), (1, 1)), mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu'),
                          b_norm=False, dense_drp=False, pad='valid', strides=((1, 1), (1, 1))):
    model = Sequential(name='Depthwise_model_classifier')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

    for i in range(len(krnl)):
        model.add(DepthwiseConv2D(kernel_size=krnl[i],
                                  depth_multiplier=depth_mul[i],
                                  activation=acts[0],
                                  padding=pad,
                                  strides=strides[i],
                                  dilation_rate=dil[i]))
        if mpool[i][0]:
            model.add(MaxPooling2D(pool_size=mpool[i]))
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
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
    return model


def depthwise_model_gap(input_shape, n_outputs, depth_mul=(4, 4), drp=0.3, krnl=((1, 3), (1, 3)), dil=((1, 1), (1, 1)),
                        mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu'), b_norm=False, dense_drp=False,
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
                                  padding=pad,
                                  dilation_rate=dil[0])

    pool = MaxPooling2D(pool_size=(1, mpool[0]))

    conv_layer2 = DepthwiseConv2D(kernel_size=(1, krnl[1]),
                                  depth_multiplier=depth_mul[1],
                                  activation=acts[0],
                                  dilation_rate=dil[1],
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


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())
