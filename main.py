import pandas as pd
from pathlib import Path
from tensorflow import keras
from DataProcessing import denormy, normed_arduino_data, arduino_data

directory = Path.cwd()

model = keras.models.load_model(directory / 'model')
def kelv_to_celsius(temp_kelv: float) -> float:
    temp_celsius = (temp_kelv) - 273.15
    return temp_celsius

print("\nARDUINO DATA\n")
print(arduino_data)

print("WEATHER PREDICTION FROM ARDUINO DATA \n")
prediction_base = normed_arduino_data


for i in range(6):
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


print('END\n ENJOY YOUR DAY')