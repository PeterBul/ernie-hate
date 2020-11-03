import os
import json
import numpy as np
import pandas as pd
from print_metrics import print_metrics

ERNIE_PATH = '../ERNIE/'
ENSAMBLE_PATH = '../configs/ensamble.json'
ARCHIVE_PATH = '../configs/archive.json'

target_names = ['Hateful', 'Offensive', 'Neither']
#target_names = ['Abusive', 'Hateful', 'Normal', 'Spam']
#target_names = ['OFF', 'NOT']


def get_file_paths(dataset):
  with open(ARCHIVE_PATH, 'r') as f:
    text = f.read()
    archive = json.loads(text)[dataset]

  return [archive[model_size][lr]['results'] for model_size in ['base', 'large'] for lr in ['1e-05', '2e-05']]

def get_preds_and_probs(dataset):
  file_paths = get_file_paths(dataset)
  preds = []
  probs = []

  for path in file_paths:
    df = pd.read_csv(os.path.join(ERNIE_PATH, path), sep="\t", names=['preds', 'probs'])
    df.probs = df.probs.apply(lambda prob: np.array([float(p) for p in prob[1:-1].split()])) 
    probs_tmp = np.stack(df.probs, axis=0)
    probs_tmp = probs_tmp[:,:2]
    probs.append(probs_tmp)
    preds.append(df.preds.to_numpy(dtype=np.int32))

  return preds, probs



"""
with open(ENSAMBLE_PATH, 'r') as f:
  text = f.read()
  file_paths = json.loads(text)[dataset]
"""


def get_majority_voting(preds):
  return [np.argmax(np.bincount(p)) for p in np.transpose(np.asarray(preds))]
  
def get_sum_of_probs(probs):
  tmp = np.asarray(probs)
  return np.argmax(np.sum(tmp, axis=0), axis=-1)

def get_y_true(dataset):
  test_df = pd.read_csv(os.path.join('../data', dataset, 'test.tsv'), sep='\t')
  return test_df.label.to_numpy()




if __name__ == "__main__":
  model_dataset = 'davidson-founta/conv'
  test_dataset = 'founta/conv'
  preds, probs = get_preds_and_probs(model_dataset)

  y_maj = get_majority_voting(preds)
  y_sum = get_sum_of_probs(probs)
  y_true = get_y_true(test_dataset)


  print("-------------Majority voting---------------")
  print_metrics(y_true, y_maj, target_names)

  print('-----------Sum of probabilities------------')
  print_metrics(y_true, y_sum, target_names)