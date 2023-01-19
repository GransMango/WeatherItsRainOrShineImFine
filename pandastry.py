import pandas as pd
import json
from IPython.display import display

with open(r"C:\Users\danie\Documents\Weathrt\WeatherItsRainOrShineImFine\Data\Oslo+Norge.json", "r") as file:
    data = json.loads(file.read())

data = data['hourly']
df = pd.DataFrame.from_dict(data)
display(df)

# Getting rows
# for data in jsonfile:
#    data_row = data['Fruit']
#    n = data['Name']
#
#    for row in data_row:
#       row['Name'] = n
#       rows.append(row)
#
# # Convert to data frame
# df = pd.DataFrame(rows)
# print(df)