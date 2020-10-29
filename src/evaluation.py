import json
import os
import numpy as np
import pandas as pd
from print_metrics import print_metrics

ERNIE_PATH = '../ERNIE/'
archive_path = '../configs/archive.json'
dataset = 'founta'
#target_names = ['Hateful', 'Offensive', 'Neither']
target_names = ['Abusive', 'Hateful', 'Normal', 'Spam']


def evaluate(model_size, lr_text):
  with open(archive_path, 'r') as f:
    text = f.read()
    archive = json.loads(text)
  path = archive[dataset][model_size][lr_text]['results']
  df = pd.read_csv(os.path.join(ERNIE_PATH, path), sep='\t', names=['preds', 'probs'])
  preds = df.preds.to_numpy(dtype=np.int32)
  test_df = pd.read_csv(os.path.join('../data', dataset, 'test.tsv'), sep='\t')
  y_true = test_df.label.to_numpy()

  print_metrics(y_true, preds, target_names)

if __name__ == "__main__":
  evaluate('large', '2e-05')