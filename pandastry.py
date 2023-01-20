import pandas as pd
import json
from apiData import json_dir


with open(json_dir, "r") as file:
    data = json.loads(file.read())

df = pd.DataFrame.from_dict(data)

print(len(df.index))

