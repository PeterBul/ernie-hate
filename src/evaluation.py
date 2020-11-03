import json
import os
import numpy as np
import pandas as pd
from print_metrics import print_metrics, get_classification_report, flatten_classification_report
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--results', type=str, default=None)
parser.add_argument('--founta2davidson', default=False, action='store_true')
parser.add_argument('--model_size', type=str)
parser.add_argument('--lr_text', type=str)

args = parser.parse_args()

ERNIE_PATH = '../ERNIE/'
archive_path = '../configs/archive.json'
model_dataset = 'founta/conv'
test_dataset = 'davidson'
#target_names = ['NOT', 'OFF']
target_names = ['Hateful', 'Offensive', 'Neither']
#target_names = ['Abusive', 'Hateful', 'Normal', 'Spam']

founta2davidson = {0:1, 1:0, 2:2, 3:2}

def evaluate(model_size, lr_text):
  with open(archive_path, 'r') as f:
    text = f.read()
    archive = json.loads(text)
  path = args.results if args.results else archive[model_dataset][test_dataset][model_size][lr_text]['results']
  df = pd.read_csv(os.path.join(ERNIE_PATH, path), sep='\t', names=['preds', 'probs'])
  
  if args.founta2davidson:
    df.preds = df.preds.apply(lambda p: founta2davidson[p])
  
  preds = df.preds.to_numpy(dtype=np.int32)
  test_df = pd.read_csv(os.path.join('../data', test_dataset, 'test.tsv'), sep='\t')
  y_true = test_df.label.to_numpy()

  print_metrics(y_true, preds, target_names)


def batch_evaluate():
  archive = load_archive()
  target_names = ['Hateful', 'Offensive', 'Neither']
  rows = []
  for test_dataset in ['davidson', 'founta/conv']:
    test_df = pd.read_csv(os.path.join('../data', test_dataset, 'test.tsv'), sep='\t')
    y_true = test_df.label.to_numpy()
    probs = []
    for model_dataset in ['davidson', 'founta/conv']:
      for model_size in ['base', 'large']:
        for lr in ['1e-05', '2e-05']:
          results_path = archive[model_dataset][model_size][lr][test_dataset]['results']
          results_df = pd.read_csv(os.path.join(ERNIE_PATH, results_path), sep='\t', names=['preds', 'probs'])
          preds = results_df.preds.to_numpy(dtype=np.int32)
          columns, row = flatten_classification_report(get_classification_report(y_true, preds, target_names))
          rows.append(row)

          results_df.probs = results_df.probs.apply(lambda prob: np.array([float(p) for p in prob[1:-1].split()])) 
          probs_tmp = np.stack(results_df.probs, axis=0)
          probs_tmp = probs_tmp[:,:2]
          probs.append(probs_tmp)
  
  return pd.DataFrame(rows, columns=columns)



def load_archive():
  with open(archive_path, 'r') as f:
    text = f.read()
  return json.loads(text)


if __name__ == "__main__":
  evaluate(args.model_size, args.lr_text)