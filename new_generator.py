import os
import random

import tushare as ts
import numpy as np
import pandas as pd

from get_tools import *
from get_samples import *


def new_generator(market='SSE', batch_size=1024, shape=4, start_date='', end_date='', lookback=61, delay=1,
                  uprate=0.0):
    # 加载权重
    # print('init generator')
    data = count_samples_weight(market, start_date=start_date, end_date=end_date, lookback=lookback,
                                delay=delay, verbose=0)
    if not data[0]:
        print('can not get data, maybe date wrong')
        return
    samples = np.zeros((batch_size,
                        lookback,
                        shape))
    targets = np.zeros((batch_size,))
    while 1:
        for i in range(batch_size):
            name = random.choices(data[0], data[1])[0]
            sample, target = get_samples_targets(ts_code=name, start_date=start_date, end_date=end_date,
                                                 lookback=lookback, delay=delay, uprate=uprate, rand=True, mod='')
            samples[i] = sample[0]
            targets[i] = target[0]
        yield samples, targets

