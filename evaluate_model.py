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

from get_samples import get_samples_targets
from get_tools import *
from make_generators import make_generators
from new_generator import new_generator


# 衡量模型对一支股票的准确率
def evaluate_old(model, ts_code='600004.SH'):
    print(ts_code)
    generator = make_generators(ts_code, shuffle=False, batch_size='auto')
    if generator is None:
        return
    result = model.evaluate_generator(generator[0], steps=1)
    print(result)


# 根据正输出的价格差衡量模型(结果为按照模型交易每个交易日平均价格变化,只适用于delay=1)
def evaluate_delta(model, ts_code='600004.SH', start_date='', end_date='', lookback=61, delay=1, base_line=0.5, verbose=1):
    data = get_samples_targets(ts_code=ts_code, start_date=start_date, end_date=end_date,
                               lookback=lookback, delay=delay, mod='delta', verbose=verbose)
    if data is None:
        return
    result = model.predict(data[0])
    result = result.T[0]
    if base_line < 0.5 and base_line != 0.0:
        predict = 1 - np.round(result - base_line + 0.5)
    else:
        predict = np.round(result - base_line + 0.5)
    return sum(predict * data[1]) / sum(predict)


# 衡量模型对一支股票的准确率
def evaluate(model, ts_code='600004.SH', start_date='', end_date='', lookback=61, delay=1):
    print(ts_code)
    data = get_samples_targets(ts_code=ts_code, start_date=start_date, end_date=end_date,
                               lookback=lookback, delay=delay)
    if data is None:
        return
    result = model.evaluate(data[0], data[1], batch_size=9999, verbose=0)
    return result


# 衡量模型对所有股票的准确率
def evaluate_total(model, market='ALL', steps=10, shape=5, start_date='', end_date='', lookback=61, delay=1, uprate=0.0):
    generator = new_generator(market=market,
                              shape=shape,
                              start_date=start_date,
                              end_date=end_date,
                              lookback=lookback,
                              delay=delay,
                              uprate=uprate,
                              batch_size=len(get_code_list(market)))
    test = next(generator)
    if test is None:
        return
    result = model.evaluate_generator(generator, steps=steps)
    return result


# 批量衡量模型对每支股票的准确率
def evaluate_all(model, market='SSE', start_date='', end_date='', lookback=61, delay=1):
    # 加载股票列表
    code_list = get_code_list(market=market)
    for code_name in code_list[:]:
        result = evaluate(model=model, ts_code=code_name, start_date=start_date, end_date=end_date,
                          lookback=lookback, delay=delay)
        print(result)


# 批量衡量模型对每支股票的delta
def evaluate_all_delta(model, market='SSE', start_date='', end_date='', lookback=61, delay=1, base_line=0.5, verbose=0):
    # 加载股票列表
    code_list = get_code_list(market=market)
    sum_list = []
    for code_name in code_list[:]:
        print_verbose(verbose, code_name)
        result = evaluate_delta(model=model, ts_code=code_name, start_date=start_date, end_date=end_date,
                                lookback=lookback, delay=delay, base_line=base_line, verbose=verbose)
        print_verbose(verbose, result)
        if result and not np.isnan(result):
            sum_list.append(result)
    print("平均:", np.average(sum_list))
    return sum_list


# 按时间衡量模型准确度
def evaluate_total_time(model, date_step=61, steps=3, start_date=20170103, end_date='', lookback=61, delay=1, uprate=0.0):
    # 计算起止index
    if start_date == '':
        start = 0
    elif int(start_date) not in date_list:
        print('can not find date')
        return
    else:
        start = date_list.index(int(start_date))
    if end_date == '':
        end = len(date_list)
    elif int(end_date) not in date_list:
        print('can not find date')
        return
    else:
        end = date_list.index(int(end_date))
    # 开始计算
    dates = []
    results = []
    for i in range(start, end, date_step):
        if i + date_step >= end:
            continue
        date = '%s : %s' % (date_list[i], date_list[i + date_step])
        print(date)
        result = evaluate_total(model, market='ALL', steps=steps, start_date=date_list[i],
                                end_date=date_list[i + date_step], lookback=lookback, delay=delay, uprate=uprate)
        if result:
            dates.append(date)
            results.append(result)
            print(result)
    plt.plot([i[2] for i in results], label='acc5', c='green')
    plt.plot([i[4] for i in results], label='acc7', c='blue')
    plt.plot([i[6] for i in results], label='acc9', c='red')
    plt.plot([i[1] for i in results], label='rec5', c='lightgreen')
    plt.plot([i[3] for i in results], label='rec7', c='lightblue')
    plt.plot([i[5] for i in results], label='rec9', c='pink')
    # plt.plot([i[8] for i in results], label='n_acc5', c='green')
    # plt.plot([i[10] for i in results], label='n_acc3', c='blue')
    # plt.plot([i[12] for i in results], label='n_acc1', c='red')
    # plt.plot([i[7] for i in results], label='n_rec5', c='lightgreen')
    # plt.plot([i[9] for i in results], label='n_rec3', c='lightblue')
    # plt.plot([i[11] for i in results], label='n_rec1', c='pink')
    plt.plot([i[13] for i in results], label='Trate', c='black')
    plt.plot([i[14] for i in results], label='Prate', c='brown')
    plt.legend()
    plt.show()
    return dates, results
