import os
import random
import time

import tushare as ts
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *

import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

from get_tools import *
from new_generator import new_generator

# GPU动态占用率
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# gen
batch_size = 1024
shape = 5
train_val_date = 20180102
val_test_date = 20191108
lookback = 61  # 244/year
delay = 1
uprate = 0.0
generator = new_generator(market='ALL', batch_size=batch_size, shape=shape,
                          start_date='', end_date=train_val_date,
                          lookback=lookback, delay=delay, uprate=uprate)
val_generator = new_generator(market='ALL', batch_size=batch_size, shape=shape,
                              start_date=train_val_date, end_date=val_test_date,
                              lookback=lookback, delay=delay, uprate=uprate)

# 建模
# *************************************** CNN ***********************************
# model = Sequential()
# kernel_size = 4
# dropout_rate = 0.3
# model.add(layers.Conv1D(8, kernel_size=kernel_size, strides=2, padding='same',
#                         input_shape=(lookback, shape)))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Conv1D(16, kernel_size=kernel_size, strides=2, padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Conv1D(32, kernel_size=kernel_size, strides=2, padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Conv1D(64, kernel_size=kernel_size, strides=2, padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Conv1D(128, kernel_size=kernel_size, strides=2, padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Conv1D(256, kernel_size=kernel_size, strides=2, padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Conv1D(512, kernel_size=kernel_size, strides=2, padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Flatten())
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=keras.optimizers.Adam(),  # lr=1e-4, epsilon=1e-8, decay=1e-4),
#               loss=keras.losses.binary_crossentropy,
#               metrics=[recall, precision, recall2, precision2, trate, prate])
# ************************************* G R U ******************************************
# dropout_rate = 0.5
# model = Sequential()
# # model.add(layers.BatchNormalization())
# model.add(layers.GRU(256,
#                      dropout=0.1,
#                      recurrent_dropout=0.5,
#                      input_shape=(None, shape)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=keras.optimizers.RMSprop(1e-4),
#               loss=keras.losses.binary_crossentropy,
#               metrics=[recall, precision, recall2, precision2, trate, prate])
# *************************************** ResNet ***************************************
# def ResBlock(x, num_filters, resampling=None, kernel_size=3):
#     def BatchActivation(x):
#         x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
#         x = Activation('relu')(x)
#         return x
#
#     def Conv(x, resampling=resampling):
#         weight_decay = 1e-4
#         if resampling is None:
#             x = Conv1D(num_filters, kernel_size=kernel_size, padding='same',
#                        kernel_initializer="he_normal",
#                        kernel_regularizer=regularizers.l2(weight_decay))(x)
#         elif resampling == 'up':
#             x = UpSampling2D()(x)
#             x = Conv1D(num_filters, kernel_size=kernel_size, padding='same',
#                        kernel_initializer="he_normal",
#                        kernel_regularizer=regularizers.l2(weight_decay))(x)
#         elif resampling == 'down':
#             x = Conv1D(num_filters, kernel_size=kernel_size, strides=2, padding='same',
#                        kernel_initializer="he_normal",
#                        kernel_regularizer=regularizers.l2(weight_decay))(x)
#         return x
#
#     a = BatchActivation(x)
#     y = Conv(a, resampling=resampling)
#     y = BatchActivation(y)
#     y = Conv(y, resampling=None)
#     if resampling is not None:
#         x = Conv(a, resampling=resampling)
#     return add([y, x])
#
#
# num_layers = int(np.log2(lookback)) - 3
# max_num_channels = lookback * 8
# weight_decay = 1e-4
#
# x_in = Input(shape=(lookback, shape))
# x = x_in
# for i in range(num_layers + 1):
#     num_channels = max_num_channels // 2 ** (num_layers - i)
#     if i > 0:
#         x = ResBlock(x, num_channels, resampling='down')
#     else:
#         x = Conv1D(num_channels, kernel_size=3, strides=2, padding='same',
#                    kernel_initializer="he_normal",
#                    kernel_regularizer=regularizers.l2(weight_decay))(x)
# x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
# x = Activation('relu')(x)
# x = GlobalAveragePooling1D()(x)
# x = Dense(1, activation='sigmoid')(x)
# model = keras.Model(x_in, x)
# model.compile(optimizer=keras.optimizers.Adam(),  # lr=1e-4, epsilon=1e-8, decay=1e-4),
#               loss=keras.losses.binary_crossentropy,
#               metrics=[recall, precision, recall2, precision2, trate, prate])
# # model.summary()
# *********************************** Attention old *********************************************
# dropout_rate = 0.3
# x_in = Input(shape=(lookback, shape))
# x = x_in
# # l = Lambda(lambda k: k[:, -1])(x)
# c = Conv1D(256, 5, activation='relu')(x)
# c = Dropout(dropout_rate)(c)
# c = Flatten()(c)
# c = Dense(lookback * shape)(c)
# c = Lambda(lambda k: K.reshape(k, (-1, lookback, shape)))(c)
# m = multiply([x, c])
# r = GRU(256)(m)
# r = Dropout(dropout_rate)(r)
# d = Dense(256)(r)
# d = Dropout(dropout_rate)(d)
# # res
# res = Dense(1, activation='sigmoid')(d)
# model = keras.Model(inputs=x_in, outputs=res)
# # model = load_model('./model/ATT140.model')
# model.compile(optimizer=keras.optimizers.Adam(),  # lr=1e-4, epsilon=1e-8, decay=1e-4),
#               loss=keras.losses.binary_crossentropy,
#               metrics=[recall, precision, recall2, precision2, trate, prate])
# *********************************** Attention *********************************************
dropout_rate = 0.3
x_in = Input(shape=(lookback, shape))
x = x_in
# x = BatchNormalization()(x)
c = Conv1D(32, 5, activation='relu')(x)
c = LeakyReLU()(c)
c = Dropout(dropout_rate)(c)
c = Flatten()(c)
c = Dense(lookback * shape)(c)
c = LeakyReLU()(c)
c = Lambda(lambda k: K.reshape(k, (-1, lookback, shape)))(c)
m = multiply([x, c])
r = GRU(256)(m)
r = LeakyReLU()(r)
r = Dropout(dropout_rate)(r)
d = Dense(256)(r)
d = LeakyReLU()(d)
d = Dropout(dropout_rate)(d)
# res
res = Dense(1, activation='sigmoid')(d)
model = keras.Model(inputs=x_in, outputs=res)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),  # lr=1e-4, epsilon=1e-8, decay=1e-4),
              loss=keras.losses.binary_crossentropy,
              metrics=[recall, precision, recall2, precision2, trate, prate]
              )
# model = load_model('./model/ATTSMALL480bad.model')
# model.load_weights('./model/ATTSMALL480bad.weight')

# callback
checkpoint = keras.callbacks.ModelCheckpoint('./model/auto_save_best.model', monitor='val_loss',
                                             verbose=1, save_best_only=True, mode='min')
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=60,
                                                            factor=0.5, min_lr=1e-8, verbose=1)
callbacks_list = [checkpoint, learning_rate_reduction]

# run
history = model.fit_generator(generator,
                              steps_per_epoch=200,  # 1min/epoch
                              epochs=480,
                              validation_data=val_generator,
                              validation_steps=10,
                              callbacks=callbacks_list,
                              # class_weight=class_weight,
                              verbose=1)

model.save('./model/auto_save.model')
model.save_weights('./model/auto_save.weight')
# show_train_history(history, 'loss', 'val_loss')
# plt.savefig('./model/auto_save.jpg')
# plt.show()
plot_history(history)

