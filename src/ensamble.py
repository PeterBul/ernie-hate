import os
import json
import numpy as np
import pandas as pd
from print_metrics import print_metrics

ERNIE_PATH = '../ERNIE/'
ENSAMBLE_PATH = '../configs/ensamble.json'
dataset = 'davidson'
target_names = ['Hateful', 'Offensive', 'Neither']

with open(ENSAMBLE_PATH, 'r') as f:
  text = f.read()
  file_paths = json.loads(text)

preds = []
probs = []

for path in file_paths:
  df = pd.read_csv(os.path.join(ERNIE_PATH, path), sep="\t", names=['preds', 'probs'])
  df.probs = df.probs.apply(lambda prob: np.array([float(p) for p in prob[1:-1].split()])) 

  probs.append(np.stack(df.probs, axis=0))
  preds.append(df.preds.to_numpy())

y_maj = [np.argmax(np.bincount(p)) for p in np.transpose(np.asarray(preds))]
y_sum = np.argmax(np.sum(np.asarray(probs), axis=0), axis=-1)


test_df = pd.read_csv(os.path.join('data', dataset, 'test.tsv'), sep='\t')

y_true = test_df.label.to_numpy()

print("-------------Majority voting---------------")
print_metrics(y_true, y_maj, target_names)

print('-----------Sum of probabilities------------')
print_metrics(y_true, y_sum, target_names)