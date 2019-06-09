from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.models import Sequential, Input, Model, load_model
from keras.engine.topology import Layer
from keras.initializers import glorot_uniform
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, GlobalAveragePooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from flipGradientTF import GradientReversal
from sklearn.metrics import accuracy_score
import numpy as np
from keras import regularizers
from keras import backend as K
import os, sys, time, json, h5py
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.initializers import RandomNormal
from keras.regularizers import l2
import functools
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, Callback
from keras.utils.training_utils import multi_gpu_model


# must choose 2 gpus
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)


def writers_adversarial_network(class_num, writers_num):
    """
    Our WSAN
    :param class_num: The total number of chinese character
    :param writers_num:
    :return:
    """
    # init softmax
    random_normal = RandomNormal(stddev=0.001, seed=1995)
    reg = 1e-9
    top5_acc = functools.partial(top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'

    inputs = Input(shape=(64, 64, 1))

    x = Conv2D(80, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(80, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    #     x = Dropout(0.1)(x)

    x = Conv2D(108, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(76, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(108, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    #     x = Dropout(0.1)(x)

    x = Conv2D(152, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(108, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(152, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    gap1 = GlobalAveragePooling2D()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    #     x = Dropout(0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    #     x = Dropout(0.2)(x)

    x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer=glorot_uniform(), use_bias=False,
               kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    gap2 = GlobalAveragePooling2D()(x)

    comb_gap = layers.Concatenate(axis=-1)([gap1, gap2])

    source_classifier = Dropout(0.5)(comb_gap)
    source_classifier = Dense(class_num, activation='softmax', name='CR',
                              kernel_initializer=random_normal, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(
        source_classifier)

    grl = GradientReversal(hp_lambda=0.01)(comb_gap)  # grl
    domain_classifier = Dropout(0.5)(grl)
    domain_classifier = Dense(writers_num, activation='softmax', name='SR',
                              kernel_initializer=random_normal, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(
        domain_classifier)

    opt = optimizers.Adadelta(lr=0.3, rho=0.95, epsilon=None, decay=1e-5)

    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    #     comb_model.summary()

    # 设置多GPU
    comb_model = multi_gpu_model(comb_model, gpus=2)
    comb_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', top5_acc])

    source_model = Model(inputs=inputs, outputs=[source_classifier])
    source_model.compile(optimizer=opt, loss='categorical_crossentropy',
                         metrics=['accuracy', top5_acc])

    domain_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_model.compile(optimizer=opt, loss='categorical_crossentropy',
                         metrics=['accuracy'])

    return comb_model, source_model, domain_model


def train_and_eval(trn_x, trn_y, trn_w, tst_x, tst_y, batch_size=128, epochs=40,
                   load_model_path=None, save_model_path=None, only_eval=False):
    """
    Note that the input does not have param: tst_w
    :return:
    """
    sample_num = len(trn_y)
    class_num = len(trn_y[0])
    writers_num = len(trn_w[0])

    comb_model, source_model, domain_model = writers_adversarial_network(class_num, writers_num)

    if load_model_path is not None:
        print('loading weight from: ', load_model_path, '...')
        comb_model.load_weights(load_model_path)

    trn_yw = np.concatenate((trn_y, trn_w), axis=1)
    tst_yw = [np.array(tst_y), np.zeros([len(tst_y), writers_num])]

    if only_eval:
        score = comb_model.evaluate(tst_x, tst_yw, batch_size=256, verbose=1)
        CR_acc = score[3]
        CR_top5_acc = score[4]
        print("Evaluate on testing set   -CR_acc: %.4f   -CR_top5_acc: %.4f" % (CR_acc, CR_top5_acc))
        return

        # Data augmentation for image
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=12,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.12,  # Randomly zoom image
        width_shift_range=0.06,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.06,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(trn_x)
    trn_gen = datagen.flow(x=trn_x, y=trn_yw, batch_size=batch_size, shuffle=True)
    format_gen = ((trn_batch_x, np.split(trn_batch_yw, [class_num, class_num + writers_num], axis=1)[:2])
                  for (trn_batch_x, trn_batch_yw) in trn_gen)

    filepath = os.path.join(save_model_path, 'model.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_CR_acc', verbose=1, save_best_only=False,
                                 save_weights_only=True, mode='auto')

    lr_reducer = ReduceLROnPlateau(monitor='val_CR_acc', factor=0.33, patience=0, verbose=1, mode='auto',
                                   min_delta=0.0001, min_lr=0.000001)

    def schedule(epoch):
        initial_lr = K.get_value(comb_model.optimizer.lr)
        if epoch == 1:
            return initial_lr * 0.1
        return initial_lr

    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
    csv_logger = CSVLogger(os.path.join(save_model_path, 'training.log'), append=False)
    callbacks_list = [checkpoint, csv_logger]  # for Adadelta
    #     callbacks_list = [checkpoint, lr_reducer, schedule, csv_logger]   # if SGD, use this to fine-tune the learning rate

    comb_model.fit_generator(format_gen, validation_data=(tst_x, tst_yw),
                             steps_per_epoch=sample_num // (batch_size), epochs=epochs, verbose=1,
                             callbacks=callbacks_list)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s subset_filepath' % sys.argv[0])
        sys.exit(1)
    op = sys.argv[1]
    hdf5_filepath = sys.argv[2]
    assert op in ['train', 'eval']

    with h5py.File(hdf5_filepath, 'r') as f:
        trn_x, trn_y, trn_w, tst_x, tst_y = f['trn/x'], f['trn/y'], f['trn/w'], f['tst/x'], f['tst/y']
        if op == 'train':
            train_and_eval(trn_x, trn_y, trn_w, tst_x, tst_y, epochs=100, batch_size=128,
                    save_model_path='model_weight')
        elif op == 'eval':
            train_and_eval(trn_x, trn_y, trn_w, tst_x, tst_y, epochs=100, batch_size=128,
                    save_model_path='model_weight', load_model_path='model_weight/model.9727.hdf5', only_eval=True)
