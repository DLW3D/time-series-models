import os
import random

import tushare as ts
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

from get_tools import *
from make_generators import make_generators

# GPU动态占用率
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# 设置token 以下方法只需要在第一次或者token失效后调用，完成调取tushare数据凭证的设置，正常情况下不需要重复设置。
# ts.set_token('ed35503ddab56a5ae561f5fd7891d2b97b6546965aa98ba2df414bc4')

# 加载股票列表
code_list = get_code_list(market='SZSE')

batch_size = 1024
gen_list = []
weight_list = []
for code in code_list[0:800]:
    print(code)
    ret = make_generators(code, batch_size=batch_size)
    if ret is not None:
        gen_list.append(ret[0])
        weight_list.append(ret[1])

val_gen_list = []
val_weight_list = []
for code in code_list[800:1000]:
    print(code)
    ret = make_generators(code, batch_size=batch_size)
    if ret is not None:
        val_gen_list.append(ret[0])
        val_weight_list.append(ret[1])

test_gen_list = []
test_weight_list = []
for code in code_list[1000:2000]:
    print(code)
    ret = make_generators(code, batch_size=batch_size)
    if ret is not None:
        test_gen_list.append(ret[0])
        test_weight_list.append(ret[1])


# 整合生成器
def cgenerator(gen_list, weight_list):
    while True:
        gen_index = random.choices(range(len(weight_list)), weight_list)[0]
        yield next(gen_list[gen_index])


cgen = cgenerator(gen_list, weight_list)
val_cgen = cgenerator(val_gen_list, val_weight_list)
test_cgen = cgenerator(test_gen_list, test_weight_list)
val_steps = 100
shape = 4


# 计算分类权重
# class_weight = {0: 1, 1: 3.195042915612453}  # lookback = 261, delay = 22, uprate = 0.10
# class_weight = {0: 1, 1: 37.71944922547332}  # lookback = 261, delay = 1, uprate = 0.10
# class_weight = {0: 1, 1: 1.0047035365638006}  # lookback = 261, delay = 1, uprate = 0.00
# class_weight = {0: 0, 1: 0}
# normal = []
# train_steps = sum(weight_list) // batch_size
# for i in range(train_steps):
#     x, y = next(cgen)
#     class_weight[0] += sum(y)
#     class_weight[1] += len(y) - sum(y)
#     # normal.append(x)
# class_weight[1] /= class_weight[0]
# class_weight[0] = 1


# 计算中值和方差
# normala = np.zeros((len(normal) * batch_size, normal[0].shape[1], normal[0].shape[2]))
# for i in range(len(normal)):
#     normala[i*batch_size:i*batch_size+batch_size] = normal[i]
# mean = normala.mean(axis=(0, 1))
# std = normala.std(axis=(0, 1))


# 正样本中有多少被识别为正样本
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    real_true = K.sum(y_true)
    return true_positives / (real_true + K.epsilon())


# 正样本中有多少被识别为正样本
def recall1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * (y_pred - 0.4), 0, 1)))
    real_true = K.sum(y_true)
    return true_positives / (real_true + K.epsilon())


# 识别为正样本中有多少是正样本
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predict_true = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predict_true + K.epsilon())


# 识别为正样本中有多少是正样本
def precision1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * (y_pred - 0.4), 0, 1)))
    predict_true = K.sum(K.round(K.clip((y_pred - 0.4), 0, 1)))
    return true_positives / (predict_true + K.epsilon())


# 加载模型时使用 keras.models.load_model(path, custom_objects=dependencies)
dependencies = {
    'recall': recall,
    'precision': precision
}

# BaseLine:recall:0.62,precision:0.26
# **************** 建模(Dense,Deep)recall:0.75,precision:0.25 (400*80不收敛)
# model = Sequential()
# model.add(layers.Flatten(input_shape=(261, shape)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy,
#               metrics=[keras.metrics.binary_accuracy, recall, precision])
#
# history = model.fit_generator(cgen,
#                               steps_per_epoch=400,  # 1min/epoch
#                               epochs=120,
#                               validation_data=val_cgen,
#                               validation_steps=val_steps,
#                               class_weight=class_weight,
#                               verbose=1)
# **************** 建模(GRU,DroupOut,Deep)recall:0.61,precision:0.27(100*250收敛)
# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.1,
#                      recurrent_dropout=0.5,
#                      return_sequences=True,
#                      input_shape=(None, shape)))
# model.add(layers.GRU(64,
#                      dropout=0.1,
#                      recurrent_dropout=0.5))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy,
#               metrics=[keras.metrics.binary_accuracy, recall, precision])
#
# history = model.fit_generator(cgen,
#                               steps_per_epoch=100,  # 1min/epoch
#                               epochs=1,
#                               validation_data=val_cgen,
#                               validation_steps=val_steps,
#                               class_weight=class_weight,
#                               verbose=1)
# **************** 建模(GRU,Dense,Deep)recall:0.6,precision:0.3(1000*500收敛)
# loss:0.62,recall0.60,precision:0.635(1500*260收敛)
# loss:0.635,recall0.67,precision:0.635(1500*350收敛)
# dropout_rate = 0.4
# model = Sequential()
# model.add(layers.CuDNNGRU(32,
#                           return_sequences=True,
#                           input_shape=(None, shape)))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.CuDNNGRU(64))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=keras.optimizers.RMSprop(1e-4),
#               loss=keras.losses.binary_crossentropy,
#               metrics=[keras.metrics.binary_accuracy, recall, precision])
#
# model = keras.models.load_model('./model/GRU35.model', custom_objects=dependencies)
# # model.load_weights('./model/cudnnGRU260.weight')
# checkpoint = keras.callbacks.ModelCheckpoint('./model/auto_save_best.model', monitor='val_loss',
#                                              verbose=1, save_best_only=True, mode='min')
# learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10,
#                                                             factor=0.5, min_lr=1e-8, verbose=1)
# callbacks_list = [checkpoint, learning_rate_reduction]
# history = model.fit_generator(cgen,
#                               steps_per_epoch=1500,  # 1min/epoch
#                               epochs=600,
#                               validation_data=val_cgen,
#                               validation_steps=val_steps,
#                               # class_weight=class_weight,
#                               callbacks=callbacks_list
#                               )
# **************** 建模(Conv,Deep)recall:0.75,precision:0.27(1000*120收敛)
# loss:0.65,recall0.63,precision:0.60(1000*960收敛)
model = Sequential()
kernel_size = 4
dropout_rate = 0.4
model.add(layers.Conv1D(8, kernel_size=kernel_size, strides=2, padding='same',
                        input_shape=(261, shape)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Conv1D(16, kernel_size=kernel_size, strides=2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Conv1D(32, kernel_size=kernel_size, strides=2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Conv1D(64, kernel_size=kernel_size, strides=2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Conv1D(128, kernel_size=kernel_size, strides=2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Conv1D(256, kernel_size=kernel_size, strides=2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Conv1D(512, kernel_size=kernel_size, strides=2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4, epsilon=1e-8, decay=1e-4),
              loss=keras.losses.binary_crossentropy,
              metrics=[recall, precision, recall1, precision1])

# model = keras.models.load_model('./model/0.75460.model', custom_objects=dependencies)
# model.save_weights('./model/0.75460.weight')
model.load_weights('./model/cnn960.weight')
checkpoint = keras.callbacks.ModelCheckpoint('./model/auto_save_best.model', monitor='val_precision',
                                             verbose=1, save_best_only=True, mode='max')
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=60,
                                                            factor=0.5, min_lr=1e-8, verbose=1)
callbacks_list = [checkpoint]
history = model.fit_generator(cgen,
                              steps_per_epoch=800,  # 1min/epoch
                              epochs=1,
                              validation_data=val_cgen,
                              validation_steps=val_steps,
                              callbacks=callbacks_list,
                              # class_weight=class_weight,
                              verbose=1)

model.save('./model/auto_save.model')
model.save_weights('./model/auto_save.weight')
plot_history(history)

# result = model.evaluate_generator(test_cgen,steps=val_steps)
