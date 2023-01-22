import pandas as pd
from os.path import dirname
from tensorflow import keras
from DataProcessing import denormy, normed_arduino_data, arduino_data, directory

model = keras.models.load_model(directory + '\\Model\\')
#print(tf.__version__)
def kelv_to_celsius(temp_kelv):
    temp_celsius = (temp_kelv) - 273.15
    return temp_celsius

print("")
print("ARDUINO DATA")
print("\n")
print(arduino_data)

print("WEATHER PREDICTION FROM ARDUINO DATA \n")
prediction_base = normed_arduino_data

print(prediction_base)
exit()

for i in range(1):
    unorm_hour_prediction = model.predict(prediction_base).flatten()
    h_prediction_r = denormy(unorm_hour_prediction)
    h_prediction_f = pd.DataFrame(h_prediction_r).T

    print(h_prediction_f)
    break


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