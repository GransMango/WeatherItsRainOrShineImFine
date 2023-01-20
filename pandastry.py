import pandas as pd
import numpy as np
import json
from IPython.display import display
from apiData import json_dir
import re

with open(json_dir, "r") as file:
    data = json.loads(file.read())

df = pd.DataFrame.from_dict(data)
dfa = df["time"].str.split("-", expand = True)
dfa.columns = ['year', 'month', 'hoursandminutes']
dfa2 = dfa["hoursandminutes"].str.split("T", expand = True)
dfa2.columns = ['trash', 'clock']
dfa3 = dfa2["clock"].str.split(":", expand = True)
dfa2.columns = ['hours', 'minutes']
#display(dfa)
dfb = df.drop(columns = "time")
dfc = dfb.join(dfa)
dfd = dfc.drop("hoursandminutes")
dfe = dfd.join(dfa3)
dff = dfe.drop("minutes")
print(list(dff.columns.values))
display(dff)
