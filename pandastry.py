import pandas as pd
import numpy as np
import json
from IPython.display import display
from apiData import json_dir
import re

with open(json_dir, "r") as file:
    data = json.loads(file.read())

df = pd.DataFrame.from_dict(data)
display(df)