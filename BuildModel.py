import os.path
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
from tensorflow.keras import layers
from DataProcessing import normed_y_train, normed_x_train, normed_x_test, normed_y_test, directory, validation_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Layers
number_of_nodes = 120 # Set amount of nodes for hidden layer 1
activation_function = 'sigmoid' # Select activation function

# Model parameters
learning_rate = 0.001 # Change learning rate
loss = 'mse' # loss function

EPOCHS = 12 # Repetitions of dataset
patience = 20 # patience of early stop


def build_model():
    model = keras.Sequential([
      layers.Dense(9, activation=activation_function, input_shape=[len(normed_x_train.keys())]), # hidden layer 1
      layers.Dense(number_of_nodes, activation=activation_function), # Hidden layer 2
      layers.Dense(9)
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
    plt.savefig(Matplot_dirname + '\\plotMAE.pdf')


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$future_temp_C^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(Matplot_dirname + '\\plotMSE.pdf')

model = build_model()
total_parameters = model.count_params()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

history = model.fit(normed_x_train, normed_y_train, epochs=EPOCHS,
                    validation_split=validation_split, verbose=0, callbacks=[early_stop, TqdmCallback(verbose=0)])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


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

if(not os.path.exists(directory + "\\ModelBenchmark")):
    os.mkdir(directory + "\\ModelBenchmark")

# Creating path for saving plots
Matplot_dirname = '_par_' + str(total_parameters) + '_lear_' + str(learning_rate) + '_' + (activation_function)
isExists = os.path.exists(directory + '\\ModelBenchmark\\' + Matplot_dirname)

# Adding mae to path, since mae shouldn't affect creation of directory
Matplot_dirname = directory + '\\ModelBenchmark\\' + 'ABS_' + str(round(mae, 2)) + Matplot_dirname

# Adding plots and a summary of parameters for model
if (not isExists):
    os.mkdir(Matplot_dirname)
    plot_history(history)
    with open(Matplot_dirname + '\\' + 'modelParam.txt', 'w+') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        file.write('Activation function: ' + activation_function + '\n')
        file.write('Learning rate: ' + str(learning_rate) + '\n')
        file.write('loss function: ' + loss + '\n')
        file.write('EPOCHS: ' + str(EPOCHS) + '\n')
        file.write('Patience: ' + str(patience) + '\n')
        file.write('Validation split: ' + str(validation_split))







