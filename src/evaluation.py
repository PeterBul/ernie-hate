import json
import os
import numpy as np
import pandas as pd
from print_metrics import print_metrics, get_classification_report, flatten_classification_report
import argparse
from ensamble import get_majority_voting, get_sum_of_probs

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
  y_true = get_y_true(test_dataset)

  print_metrics(y_true, preds, target_names)

def get_path(model_dataset, test_dataset, model_size, lr):
  archive = load_archive()
  return archive[model_dataset][test_dataset][model_size][lr]['results']

def batch_evaluate():
  rows = []
  y_true = get_y_true(test_dataset)
  probs = []
  preds = []
  for model_size in ['base', 'large']:
    for lr in ['1e-05', '2e-05']:
      preds_tmp, probs_tmp = get_preds_and_probs(model_dataset, test_dataset, model_size, lr)
      row = flatten_classification_report(get_classification_report(y_true, preds_tmp, target_names), name="{} {}".format(model_size, lr))
      rows.append(row)
      probs.append(probs_tmp)
      preds.append(preds_tmp)
  
  y_maj = get_majority_voting(preds)
  y_sum = get_sum_of_probs(probs)
  row = flatten_classification_report(get_classification_report(y_true, y_maj, target_names), name="Ensamble Majority Vote")
  rows.append(row)
  columns, row = flatten_classification_report(get_classification_report(y_true, y_sum, target_names), name="Ensamble Sum", return_columns=True)
  rows.append(row)
  return pd.DataFrame(rows, columns=columns)

def get_preds_and_probs(model_dataset, test_dataset, model_size, lr):
  path = get_path(model_dataset, test_dataset, model_size, lr)
  df = pd.read_csv(os.path.join(ERNIE_PATH, path), sep="\t", names=['preds', 'probs'])
  df.probs = df.probs.apply(lambda prob: np.array([float(p) for p in prob[1:-1].split()])) 
  probs = np.stack(df.probs, axis=0)
  probs = probs[:,:len(target_names)]
  preds = df.preds.to_numpy(dtype=np.int32)
  return preds, probs


def get_y_true(dataset):
  test_df = pd.read_csv(os.path.join('../data', dataset, 'test.tsv'), sep='\t')
  return test_df.label.to_numpy()



def load_archive():
  with open(archive_path, 'r') as f:
    text = f.read()
  return json.loads(text)


if __name__ == "__main__":
  evaluate(args.model_size, args.lr_text)