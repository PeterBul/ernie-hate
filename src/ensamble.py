import os
import json
import pandas as pd

ERNIE_PATH = '../ERNIE/'
ENSAMBLE_PATH = '../configs/ensamble.json'

with open(ENSAMBLE_PATH, 'r') as f:
  text = f.read()
  file_paths = json.loads(text)


