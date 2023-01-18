import pandas as pd
from os.path import dirname
from tensorflow import keras

directory = dirname(__file__)

model = keras.models.load_model(directory + '\\Model\\')
#print(tf.__version__)
def kelv_to_celsius(temp_kelv):
    temp_celsius = (temp_kelv) - 273.15
    return temp_celsius

arduino_da = pd.read_csv(directory + '\\Data\\arduino_data.csv')
arduino_da['temp_C'] = kelv_to_celsius((arduino_da['temp']))
arduino_features = ['temp_C', 'pressure', 'humidity']
arduino_data = pd.DataFrame(arduino_da, columns=arduino_features)



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

#Train x stats og y stats

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


#Trenger bare denormy i main, men normx og normy i buildmodel
# Lagrer denormy i en variabel ved å kalle på en annen fil.
def normx(x):
  return (x - train_x_stats['mean']) / train_x_stats['std']
def normy(x):
  return (x - train_y_stats['mean']) / train_y_stats['std']
def denormy(x):
    return (x * train_y_stats['std']) + train_y_stats['mean']

normed_arduino_data = normx(arduino_data)


print("")
print("ARDUINO DATA")
print("\n")
print(arduino_data)

print("WEATHER PREDICTION FROM ARDUINO DATA \n")
prediction_base = normed_arduino_data


for i in range(5):
    unorm_hour_prediction = model.predict(prediction_base).flatten()
    h_prediction_r = denormy(unorm_hour_prediction)
    h_prediction_f = pd.DataFrame(h_prediction_r).T
    hour_prediction = h_prediction_f.rename(columns={1: 'pressure',
                                                     2: 'humidity',
                                                     0: 'temp_C',
                                                     3: 'rain'})
    hour_prediction.loc[hour_prediction['rain'] < 0, 'rain'] = 0
    df_hd = pd.DataFrame(data=unorm_hour_prediction).T
    df_h = pd.DataFrame(df_hd, columns={0, 1, 2})
    prediction_base = df_h
    print("\n")
    print(str(i + 1) + ' HOUR')
    print(round(hour_prediction, 2))


print('END')
print('ENJOY YOUR DAY')