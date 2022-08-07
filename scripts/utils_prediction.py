import os
import numpy as np
import random
import tensorflow as tf
from cond_rnn import ConditionalRNN
from keras.models import Sequential
from keras.layers import Dense
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sktime.datatypes._panel._convert import from_3d_numpy_to_multi_index
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN
import pandas as pd

from scripts.utils import PROCESSED_PATH

N_SPLITS_OUTER = 10
N_SPLITS_INNER = 5
VERBOSE = 0
LONGI = ['TimePP',
         'SpO2',
         'FiO2',
         'RR',
         # 'SF'
         # 'ROX'
         ]
STATIC = True
STATIC_BINARY = ['X04_Gender']
# 'X22_CCD',
# 'X24_DM',
# 'X25_CKD',
# 'X26_Cancer',
# 'X27_Obesity',
# 'X28_CLF',
# 'X30_Steroid'
# ]
STATIC_CONTINUOUS = ['X05_Age',
                     'Nb_Comorbidities',
                     'X44b_Delta_Post_PP_RR',
                     'X45a_Delta_Post_PP_SpO2',
                     'X45b_Delta_Post_PP_FiO2',
                     # 'X48e_Delta_Post_PP_ROX',
                     # 'X33e_Delta_Post_PP_SF',
                     ]

PARAM_GRID = {
    'num_units': [16, 32, 64],
    'lr': [0.1, 0.01],  # Common values
    'epochs': 100,
    'batch_size': [25, 50],
    'reccurrent_dropout': [0, 0.1]  # Between 0 and 0.1
}


def load_data_nan():
    # Returns inputs and targets dataframes (NaN values included)
    x = pd.read_pickle(os.path.join(PROCESSED_PATH, 'inputs_nan.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_PATH, 'targets_nan.pkl'))
    return x, y


def binarize_array(x):
    """One Hot Encoding features in 3 categories Low/Medium/High
    x must be a standardized array of time series (n_ts, n_timesteps, n_features)"""
    copy = x.reshape(-1, x.shape[-1])
    new_x = np.empty((copy.shape[0], copy.shape[1] * 3))
    new_x.fill(0)
    i = 0
    while i < copy.shape[1]:
        new_x[:, 3 * i] = (copy[:, i] <= -1).astype(float)
        new_x[:, 3 * i + 1] = ((copy[:, i] > -1) & (copy[:, i] <= 1)).astype(float)
        new_x[:, 3 * i + 2] = (copy[:, i] > 1).astype(float)
        i += 1
    new_x = new_x.reshape((x.shape[0], x.shape[1], new_x.shape[1]))
    return new_x


def padding(x):
    df = from_3d_numpy_to_multi_index(np.swapaxes(x, 1, 2))
    for index in df.index.get_level_values('instances').unique():
        df.loc[index, :].ffill(inplace=True)
        df.loc[index, :].bfill(inplace=True)
    n_samples = len(df.index.get_level_values('instances').unique())
    n_timesteps = len(df.index.get_level_values('timepoints').unique())
    n_features = len(df.columns)
    return df.values.reshape((n_samples, n_timesteps, n_features))


def scale_process_inputs(x_train, c2_train, x_test, c2_test, binarization):
    # Process inputs
    if binarization:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
        c2_train = scaler.fit_transform(c2_train)
        c2_test = scaler.transform(c2_test)
        x_train = binarize_array(x_train)
        x_test = binarize_array(x_test)
    else:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
        c2_train = scaler.fit_transform(c2_train)
        c2_test = scaler.transform(c2_test)
        # x_train = padding(x_train)
        # x_test = padding(x_test)
        x_train = np.nan_to_num(x_train, nan=-4)
        x_test = np.nan_to_num(x_test, nan=-4)
    return x_train, c2_train, x_test, c2_test


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, x, y=None):
        self.scaler.fit(x.reshape(-1, x.shape[-1]))
        return self

    def transform(self, x, y=None):
        x = self.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x


def set_seed(seed):
    """Fix seed for reproducibility"""
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_rnn(param):
    model = Sequential()
    model.add(SimpleRNN(param['num_units'], recurrent_dropout=param['reccurrent_dropout'], return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))  # Output layer, sigmoid to output probabilities.
    # 'accuracy' transformed to BinaryAccuracy with default threshold=0.5
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=param['lr']), metrics=['accuracy'])
    return model


def get_condrnn(param):
    model = Sequential()
    model.add(ConditionalRNN(param['num_units'], cell='RNN', recurrent_dropout=param['reccurrent_dropout'],
                             return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))  # Output layer, sigmoid to output probabilities.
    # 'accuracy' transformed to BinaryAccuracy with default threshold=0.5
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=param['lr']), metrics=['accuracy'])
    return model
