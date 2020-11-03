import os
import json
import numpy as np
import pandas as pd
from print_metrics import print_metrics

ERNIE_PATH = '../ERNIE/'
ENSAMBLE_PATH = '../configs/ensamble.json'
ARCHIVE_PATH = '../configs/archive.json'
model_dataset = 'davidson-founta/conv'
test_dataset = 'founta/conv'
target_names = ['Hateful', 'Offensive', 'Neither']
#target_names = ['Abusive', 'Hateful', 'Normal', 'Spam']
#target_names = ['OFF', 'NOT']

with open(ARCHIVE_PATH, 'r') as f:
  text = f.read()
  archive = json.loads(text)[model_dataset]

file_paths = [archive[model_size][lr]['results'] for model_size in ['base', 'large'] for lr in ['1e-05', '2e-05']]

"""
with open(ENSAMBLE_PATH, 'r') as f:
  text = f.read()
  file_paths = json.loads(text)[dataset]
"""


preds = []
probs = []

for i, path in enumerate(file_paths):
  df = pd.read_csv(os.path.join(ERNIE_PATH, path), sep="\t", names=['preds', 'probs'])
  df.probs = df.probs.apply(lambda prob: np.array([float(p) for p in prob[1:-1].split()])) 
  probs_tmp = np.stack(df.probs, axis=0)
  probs_tmp = probs_tmp[:,:3]
  print(probs_tmp.dtype)
  probs.append(probs_tmp)
  preds.append(df.preds.to_numpy(dtype=np.int32))

y_maj = [np.argmax(np.bincount(p)) for p in np.transpose(np.asarray(preds))]
for prob in probs:
  print(prob.shape)
tmp = np.asarray(probs)
y_sum = np.argmax(np.sum(tmp, axis=0), axis=-1)
test_df = pd.read_csv(os.path.join('../data', test_dataset, 'test.tsv'), sep='\t')
y_true = test_df.label.to_numpy()

print("-------------Majority voting---------------")
print_metrics(y_true, y_maj, target_names)

print('-----------Sum of probabilities------------')
print_metrics(y_true, y_sum, target_names)