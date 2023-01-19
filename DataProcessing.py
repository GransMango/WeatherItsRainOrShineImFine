import os

import pandas as pd
from os.path import dirname

def kelv_to_celsius(temp_kelv):
    temp_celsius = (temp_kelv) - 273.15
    return temp_celsius

directory = dirname(__file__)


da = pd.read_csv(directory + '\\Data\\bradatebackup.csv')
# da["temp_C"] = kelv_to_celsius(da["temp"])
features = ['temp_C', 'pressure', 'humidity', 'rain']
dataset = pd.DataFrame(da, columns=features)


x_dataset = pd.DataFrame(dataset, columns=['temp_C', 'pressure', 'humidity'])
x_train = x_dataset.loc[0:12425]
x_test = x_dataset.loc[12426:13030]


y_datasetdf = dataset.shift(periods=-1)
y_dataset = pd.DataFrame(y_datasetdf, columns=['temp_C', 'pressure', 'humidity', 'rain'])
y_train = y_dataset.loc[0:12425]
y_test = y_dataset.loc[12426:13030]


train_x_stats = x_train.describe().transpose()
train_y_stats = y_train.describe().transpose()



def normx(x):
  return (x - train_x_stats['mean']) / train_x_stats['std']
def normy(x):
  return (x - train_y_stats['mean']) / train_y_stats['std']
def denormy(x):
    return (x * train_y_stats['std']) + train_y_stats['mean']

normed_x_train = normx(x_train)
normed_x_test = normx(x_test)

normed_y_train = normy(y_train)
normed_y_test = normy(y_test)

arduino_da = pd.read_csv(directory + '\\Data\\arduino_data.csv')
arduino_da['temp_C'] = kelv_to_celsius((arduino_da['temp']))
arduino_features = ['temp_C', 'pressure', 'humidity']
arduino_data = pd.DataFrame(arduino_da, columns=arduino_features)

normed_arduino_data = normx(arduino_data)
