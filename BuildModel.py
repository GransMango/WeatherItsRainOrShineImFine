import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os.path import dirname

def kelv_to_celsius(temp_kelv):
    temp_celsius = (temp_kelv) - 273.15
    return temp_celsius

directory = dirname(__file__)

#print("DATASET DATAFRAME")
da = pd.read_csv(directory + '\\Data\\bradatebackup.csv')
da["temp_C"] = kelv_to_celsius(da["temp"])
features = ['temp_C', 'pressure', 'humidity', 'rain']
dataset = pd.DataFrame(da, columns=features)


#print(dataset)

#print("x_train")
x_dataset = pd.DataFrame(dataset, columns=['temp_C', 'pressure', 'humidity'])
x_train = x_dataset.loc[0:12425]
x_test = x_dataset.loc[12426:13030]
#print(x_train)
#print("x_test")
#print(x_test)

#print("y_train")
y_datasetdf = dataset.shift(periods=-1)
y_dataset = pd.DataFrame(y_datasetdf, columns=['temp_C', 'pressure', 'humidity', 'rain'])
y_train = y_dataset.loc[0:12425]
y_test = y_dataset.loc[12426:13030]
#print(y_train)
#print("y_test")
#print(y_test)

train_x_stats = x_train.describe().transpose()
#print(train_x_stats)
train_y_stats = y_train.describe().transpose()
#print(train_y_stats)


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

#print("normed x_test")
#print(normed_x_test)
#print("normed x_train")
#print(normed_x_train)
#print("normed y_test")
#print(normed_y_test)
#print("normed y_train")
#print(normed_y_train)

#print("MODEL SUMMARY")
def build_model():
  model = keras.Sequential([
    layers.Dense(5, activation='sigmoid', input_shape=[len(x_train.keys())]), # hidden layer 1
    layers.Dense(4, activation='sigmoid'), # Hidden layer 2
    layers.Dense(4)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001) # learning rate i parentes
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']) # Mean Absolute Error & Root Mean Squared Error
  return model
model = build_model()
#print(model.summary())

EPOCHS = 5000 # endre antall runder den går gjennom hele datasettet

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(normed_x_train, normed_y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [future_temp_C]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$future_temp_C^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()
  plt.show()

#plot_history(history)

#loss, mae, mse = model.evaluate(normed_x_test, normed_y_test, verbose=2)
#print("Testing set Mean Abs Error: {:5.2f} norm fut weather".format(mae))

#print('WEIGHTS VALUES')
#for layerNum, layer in enumerate(model.layers):
#    weights = layer.get_weights()[0]
#    for fromNeuronNum, wgt in enumerate(weights):
#        for toNeuronNum, wgt2 in enumerate(wgt):
#            print(f'L{layerNum}N{fromNeuronNum} -> L{layerNum+1}N{toNeuronNum} = {wgt2}')

model.save(directory + '\\Model\\')