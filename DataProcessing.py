import json
from apiData import json_dir
import pandas as pd
from os.path import dirname

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
dataset_length = len(da.index)
validation_split = 0.2 # validation data percentage
trainsplit = dataset_length*(1-validation_split)

x_dataset = da
x_train = x_dataset
x_test = x_dataset.loc[trainsplit:dataset_length]

y_dataset = da.drop(["hour", "day", "month"])
y_dataset = da.shift(periods=-1)
y_train = y_dataset
y_test = y_dataset.loc[trainsplit:dataset_length]


train_x_stats = x_train.describe().transpose()
train_y_stats = y_train.describe().transpose()

normed_x_train = normx(x_train)
normed_x_test = normx(x_test)

normed_y_train = normy(y_train)
normed_y_test = normy(y_test)

arduino_da = pd.read_csv(directory + '\\Data\\arduino_data.csv')
arduino_features = ['temp_C', 'pressure', 'humidity']
arduino_data = pd.DataFrame(arduino_da, columns=arduino_features)
normed_arduino_data = normx(arduino_data)
