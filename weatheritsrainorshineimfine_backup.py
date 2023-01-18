import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#print(tf.__version__)
def kelv_to_celsius(temp_kelv):
    temp_celsius = (temp_kelv) - 273.15
    return temp_celsius

arduino_da = pd.read_csv(r'C:\Users\nik1106\Documents\Skolearbeid\ToF\arduino_data.csv')
arduino_da['temp_C'] = kelv_to_celsius((arduino_da['temp']))
arduino_features = ['temp_C', 'pressure', 'humidity']
arduino_data = pd.DataFrame(arduino_da, columns=arduino_features)

#print("DATASET DATAFRAME")
da = pd.read_csv(r'C:\Users\nik1106\Documents\Skolearbeid\ToF\bradate.csv')
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

train_x_stats = x_train.describe()
train_x_stats = train_x_stats.transpose()
#print(train_x_stats)
train_y_stats = y_train.describe()
train_y_stats = train_y_stats.transpose()
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

normed_arduino_data = normx(arduino_data)

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

print("")
print("ARDUINO DATA")
print(arduino_data)

print("WEATHER PREDICTION FROM ARDUINO DATA")
prediction_base = normed_arduino_data
unorm_one_h_prediction = model.predict(prediction_base).flatten()
one_h_prediction_r = denormy(unorm_one_h_prediction)
one_h_prediction_f = pd.DataFrame(one_h_prediction_r).T
one_h_prediction = one_h_prediction_f.rename(columns={1:'pressure',
                                                      2:'humidity',
                                                      0:'temp_C',
                                                      3:'rain'})
one_h_prediction.loc[one_h_prediction['rain'] < 0, 'rain'] = 0
df_onehd = pd.DataFrame(data=unorm_one_h_prediction).T
df_oneh = pd.DataFrame(df_onehd, columns={0,1,2})
print('1 HOUR')
print(round(one_h_prediction, 2))

unorm_two_h_prediction = model.predict(df_oneh).flatten()
two_h_prediction_r = denormy(unorm_two_h_prediction)
two_h_prediction_f = pd.DataFrame(two_h_prediction_r).T
two_h_prediction = two_h_prediction_f.rename(columns={1:'pressure',
                                                      2:'humidity',
                                                      0:'temp_C',
                                                      3:'rain'})
two_h_prediction.loc[two_h_prediction['rain'] < 0, 'rain'] = 0
df_twohd = pd.DataFrame(data=unorm_two_h_prediction).T
df_twoh = pd.DataFrame(df_twohd, columns={0,1,2})
print('2 HOUR')
print(round(two_h_prediction,2))

unorm_three_h_prediction = model.predict(df_twoh).flatten()
three_h_prediction_r = denormy(unorm_three_h_prediction)
three_h_prediction_f = pd.DataFrame(three_h_prediction_r).T
three_h_prediction = three_h_prediction_f.rename(columns={1:'pressure',
                                                          2:'humidity',
                                                          0:'temp_C',
                                                          3:'rain'})
three_h_prediction.loc[three_h_prediction['rain'] < 0, 'rain'] = 0
df_threehd = pd.DataFrame(data=unorm_three_h_prediction).T
df_threeh = pd.DataFrame(df_threehd, columns={0,1,2})
print('3 HOUR')
print(round(three_h_prediction,2))

unorm_four_h_prediction = model.predict(df_threeh).flatten()
four_h_prediction_r = denormy(unorm_four_h_prediction)
four_h_prediction_f = pd.DataFrame(four_h_prediction_r).T
four_h_prediction = four_h_prediction_f.rename(columns={1:'pressure',
                                                        2:'humidity',
                                                        0:'temp_C',
                                                        3:'rain'})
four_h_prediction.loc[four_h_prediction['rain'] < 0, 'rain'] = 0
df_fourhd = pd.DataFrame(data=unorm_four_h_prediction).T
df_fourh = pd.DataFrame(df_fourhd, columns={0,1,2})
print('4 HOUR')
print(round(four_h_prediction,2))

unorm_five_h_prediction = model.predict(df_fourh).flatten()
five_h_prediction_r = denormy(unorm_five_h_prediction)
five_h_prediction_f = pd.DataFrame(five_h_prediction_r).T
five_h_prediction = five_h_prediction_f.rename(columns={1:'pressure',
                                                        2:'humidity',
                                                        0:'temp_C',
                                                        3:'rain'})
five_h_prediction.loc[five_h_prediction['rain'] < 0, 'rain'] = 0
df_fivehd = pd.DataFrame(data=unorm_five_h_prediction).T
df_fiveh = pd.DataFrame(df_fivehd, columns={0,1,2})
print('5 HOUR')
print(round(five_h_prediction, 2))

unorm_six_h_prediction = model.predict(df_fiveh).flatten()
six_h_prediction_r = denormy(unorm_six_h_prediction)
six_h_prediction_f = pd.DataFrame(six_h_prediction_r).T
six_h_prediction = six_h_prediction_f.rename(columns={1:'pressure',
                                                      2:'humidity',
                                                      0:'temp_C',
                                                      3:'rain'})
six_h_prediction.loc[six_h_prediction['rain'] < 0, 'rain'] = 0
df_sixhd = pd.DataFrame(data=unorm_six_h_prediction).T
df_sixh = pd.DataFrame(df_sixhd, columns={0,1,2})
print('6 HOUR')
print(round(five_h_prediction,2))



print('END')
print('ENJOY YOUR DAY')