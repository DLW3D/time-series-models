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

from get_samples import get_samples
from get_tools import *


# 搜索预测值高的股票
def search_predict(model, date=20191128, market='SSE', duiring=1, baseline=0.9, verbose=1):
    # 加载股票列表
    code_list = get_code_list(market=market)
    # 准备循环
    sum_pred = None
    sum_count = 0
    result_code = []
    result_pred = []
    for code_name in code_list[:]:
        print_verbose(verbose, code_name)
        samples = get_samples(ts_code=code_name, date=date, duiring=duiring, verbose=verbose)
        if samples is None:
            continue
        # 统计
        pred = model.predict(samples)
        if sum_count == 0:
            sum_pred = np.round(pred)
        else:
            sum_pred = sum_pred + np.round(pred)
        sum_count += 1
        # 判断
        if any(pred > baseline):
            result_code.append(code_name)
            result_pred.append(pred)
            print_verbose(verbose, '%s*****************************************' % code_name)
            print_verbose(verbose, pred)
    rate_pred = sum_pred / sum_count
    print('rate_pred:\n%s' % rate_pred)
    for i in range(len(result_code)):
        print('%s*****************************************\n%s' % (result_code[i], result_pred[i]))
    return result_code, result_pred, rate_pred
