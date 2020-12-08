import pandas as pd
from tqdm import tqdm
import numpy as np
from tokenization import Processor

tqdm.pandas()
mapping = {0:1, 1:0, 2:2, 3:2}

def convert_dataset(ds):
  df = pd.read_csv('../data/founta/{}.tsv'.format(ds), sep='\t')

  df.label = df.label.apply(lambda l: mapping[l])

  print(df.head())

  df.to_csv('../data/founta/conv/{}.tsv'.format(ds), sep='\t', index=False)

def turn_labels():
  mapping2 = {0:1, 1:2, 2:2, 3:3}
  for ds in ['train', 'dev', 'test', 'train-full']:
    df = pd.read_csv('../data/founta/isaksen/{}.tsv'.format(ds), sep='\t')
    df.label = df.label.progress_apply(lambda l: mapping[l])
    df.to_csv('../data/founta/isaksen/{}.tsv'.format(ds), sep='\t', index=False)
  
  for ds in ['train', 'dev', 'test', 'train-full']:
    df = pd.read_csv('../data/founta/isaksen/spam/{}.tsv'.format(ds), sep='\t')
    df.label = df.label.progress_apply(lambda l: mapping[l])
    df.to_csv('../data/founta/isaksen/spam/{}.tsv'.format(ds), sep='\t', index=False)


def remove_neither(ds):
  df = pd.read_csv('../data/davidson/{}.tsv'.format(ds), sep='\t')
  df = df[df.label != 2]
  print(df.label.value_counts())
  print(df.head())

  df.to_csv('../data/davidson/filtered/{}.tsv'.format(ds), sep='\t', index=False)

def get_train_dev_test_dataframes(df, train_fraction, dev_fraction, test_fraction):
  """Gets a collection for the training, dev and test set"""
  if dev_fraction > 0 and test_fraction > 0:
    train, dev, test = np.split(df.sample(frac=1, random_state=42), 
                                [int(train_fraction*len(df)), 
                                  int((train_fraction + dev_fraction)*len(df))])
    print("Train: {}, dev: {}, test: {}".format(len(train), len(dev), len(test)))
  else:
    raise ValueError("The processor doesn't support not using dev and test set")
  return train, dev, test

def preprocess_founta(convert=False):
  processor = Processor(preserve_case=False, reduce_len=False, strip_handles=False, demojize=True, replace_url=True, segment_hashtags=True, correct_user=True, url_to_http=True, remove_rt=True, change_at=False)
  read_path = 'data/founta/founta_spam.csv'
  df = pd.read_csv(read_path)
  df.tweet = df.tweet.progress_apply(processor.process)
  if convert:
    df.label = df.label.progress_apply(lambda x: mapping[x])
  
  df = df.rename(columns={'tweet': 'text_a'})

  train, dev, test = get_train_dev_test_dataframes(df, 0.6, 0.2, 0.2)
  full = pd.concat([train, dev])
  for string, df in zip(['train', 'dev', 'test', 'train-full'], [train, dev, test, full]):
    save_path = '../data/founta/isaksen/spam/{}.tsv'.format(string)
    df.to_csv(save_path, sep='\t', index=False)
  



def oversample(ds):
  df = pd.read_csv('../data/davidson/filtered/{}.tsv'.format(ds), sep='\t')
  new_df = pd.DataFrame({'text_a':[], 'label': []})
  for _, row, in tqdm(df.iterrows(), total=df.shape[0]):
    new_df = new_df.append(row, ignore_index=True)
    if row.label == 0:
      for i in range(7):
        new_df = new_df.append(row, ignore_index=True)
  new_df.label = new_df.label.astype(int)
  new_df = new_df.sample(frac=1, random_state=42)
  new_df.to_csv('../data/davidson/filtered/{}-oversampled.tsv'.format(ds), sep='\t', index=False)

def combine():
  d_train = pd.read_csv('../data/davidson/train-full.tsv', sep='\t')
  f_train = pd.read_csv('../data/founta/isaksen/train-full.tsv', sep='\t')
  s_train = pd.read_csv('../data/solid/conv/dev.tsv', sep='\t')
  
  c_train = pd.concat([d_train, f_train, s_train])
  del d_train, f_train, s_train

  c_train.to_csv('../data/combined2/train.tsv', sep='\t', index=False)
  del c_train
  
  test = []
  for ds in ['davidson', 'founta/isaksen', 'solid/conv']:
    test.append(pd.read_csv('../data/{}/test.tsv'.format(ds), sep='\t'))
  
  c_test = pd.concat(test)
  del test

  c_test.to_csv('../data/combined2/test.tsv', sep='\t', index=False)
    


combine()
