import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os.path import dirname
from DataProcessing import x_train, normed_y_train, normed_x_train

directory = dirname(__file__)

def build_model():
    model = keras.Sequential([
      layers.Dense(3, activation='sigmoid', input_shape=[len(x_train.keys())]), # hidden layer 1
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

plot_history(history)

#loss, mae, mse = model.evaluate(normed_x_test, normed_y_test, verbose=2)
#print("Testing set Mean Abs Error: {:5.2f} norm fut weather".format(mae))

#print('WEIGHTS VALUES')
#for layerNum, layer in enumerate(model.layers):
#    weights = layer.get_weights()[0]
#    for fromNeuronNum, wgt in enumerate(weights):
#        for toNeuronNum, wgt2 in enumerate(wgt):
#            print(f'L{layerNum}N{fromNeuronNum} -> L{layerNum+1}N{toNeuronNum} = {wgt2}')


# Saving model file
model.save(directory + '\\Model\\')
