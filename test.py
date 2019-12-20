import time
import os
import shutil
import tensorflow as tf

from load_tools import *
from get_tools import *
from get_samples import *
from new_generator import *
from evaluate_model import *
from history_predict import *
from serch_predict import *

# 加载模型
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
model = load_model(model_name='./model/binary/ATT140to740.model')

evaluate(model)
dates, results = evaluate_total_time(model, date_step=22, steps=1, start_date=20191008, end_date='', delay=1)
sum_list = evaluate_all_delta(model, market='SSE', start_date=20191108, end_date='', base_line=0.5, delay=1)
# result_code, result_pred, rate_pred = search_predict(model, date=20190102, duiring=1, market='ALL')
history_predict(model, mod='complex', delay=1)
