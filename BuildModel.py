import keras
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from DataProcessing import x_train, normed_y_train, normed_x_train

directory = Path.cwd()

# Function to build the neural network model
def build_model() -> keras.Model:
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

# Define the number of epochs for training
EPOCHS = 5000

# Custom callback to print progress during training
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 100 == 0: print('')
      print('.', end='')


# Early stopping callback to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Train the model
history = model.fit(normed_x_train, normed_y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

# Plot training history
def plot_history(history) -> None:
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

# Saving model file
model.save(directory / 'Model')