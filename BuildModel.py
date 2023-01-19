import os.path
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DataProcessing import x_train, normed_y_train, normed_x_train, normed_x_test, normed_y_test, directory

# Layers
number_of_nodes = 120 # Set amount of nodes for hidden layer 1
activation_function = 'sigmoid' # Select activation function

# Model parameters
learning_rate = 0.001 # Change learning rate
loss = 'mse' # loss function

EPOCHS = 5000 # Repetitions of dataset
patience = 20 # patience of early stop
validation_split = 0.2 # validation data percentage

def build_model():
    model = keras.Sequential([
      layers.Dense(3, activation=activation_function, input_shape=[len(x_train.keys())]), # hidden layer 1
      layers.Dense(number_of_nodes, activation=activation_function), # Hidden layer 2
      layers.Dense(4)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mae', 'mse']) # Mean Absolute Error & Root Mean Squared Error
    return model
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
    plt.savefig(directory + "\\ModelBenchmark\\" + "test2.pdf")


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$future_temp_C^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(directory + "\\ModelBenchmark\\" + "test.pdf")


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 100 == 0: print('')
      print('.', end='')


model = build_model()
model.summary()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

history = model.fit(normed_x_train, normed_y_train, epochs=EPOCHS,
                    validation_split=validation_split, verbose=2, callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch



plot_history(history)

# loss, mae, mse = model.evaluate(normed_x_test, normed_y_test, verbose=2)
mae = model.evaluate(normed_x_test, normed_y_test, verbose=2)[1]
print("Testing set Mean Abs Error: {:5.2f} norm fut weather".format(mae))

#print('WEIGHTS VALUES')
#for layerNum, layer in enumerate(model.layers):
#    weights = layer.get_weights()[0]
#    for fromNeuronNum, wgt in enumerate(weights):
#        for toNeuronNum, wgt2 in enumerate(wgt):
#            print(f'L{layerNum}N{fromNeuronNum} -> L{layerNum+1}N{toNeuronNum} = {wgt2}')


# Saving model file
model.save(directory + '\\Model\\')
#isExists = os.path.exists(directory + '\\ModelBenchmark\\ABS_' + str(mae) + '_' + )
#with open(directory + '\\ModelBenchmark\\ABS_' + str(mae) + )
