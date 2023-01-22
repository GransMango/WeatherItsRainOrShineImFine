import json
from apiData import json_dir
import pandas as pd
from os.path import dirname
import numpy as np

def normx(x):
  return (x - train_x_stats['mean']) / train_x_stats['std']
def normy(x):
  return (x - train_y_stats['mean']) / train_y_stats['std']
def denormy(x):
    return (x * train_y_stats['std']) + train_y_stats['mean']

directory = dirname(__file__)

with open(json_dir, "r") as file:
    data = json.loads(file.read())

da = pd.DataFrame.from_dict(data)
dataset_length = len(da.index) - 2
validation_split = 0.2 # validation data percentage
trainsplit = dataset_length*(1-validation_split)

x_dataset = da
x_train = x_dataset.loc[0:(dataset_length)]
x_test = x_dataset.loc[trainsplit:dataset_length]
#x_train= np.asarray(x_train).astype(np.int)


#y_dataset = da.drop(["hour", "day", "month"]).shift(periods=-1)
y_dataset = da.shift(periods=-1)
y_train = y_dataset.loc[0:(dataset_length)]
y_test = y_dataset.loc[trainsplit:dataset_length]
#y_train= np.asarray(y_train).astype(np.int)

train_x_stats = x_train.describe().transpose()
train_y_stats = y_train.describe().transpose()

normed_x_train = normx(x_train)
normed_x_test = normx(x_test)

normed_y_train = normy(y_train)
normed_y_test = normy(y_test)

with open(directory + '\\Data\\arduino_data.json', "r") as file:
    a_data = json.loads(file.read())

arduino_features = ['temperature_2m', 'relativehumidity_2m', 'pressure_msl', 'precipitation', 'windspeed_10m', 'winddirection_10m', 'month', 'day', 'hour']
arduino_da = pd.DataFrame.from_dict(a_data)
arduino_data = pd.DataFrame(arduino_da, columns=arduino_features)
normed_arduino_data_raw = normx(arduino_da)
normed_arduino_data = pd.DataFrame(normed_arduino_data_raw, columns=arduino_features)

print(normed_x_train.keys())
print(normed_y_test)

