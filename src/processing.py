import pandas as pd
from tqdm import tqdm

mapping = {0:1, 1:0, 2:2, 3:2}

def convert_dataset(ds):
  df = pd.read_csv('../data/founta/{}.tsv'.format(ds), sep='\t')

  df.label = df.label.apply(lambda l: mapping[l])

  print(df.head())

  df.to_csv('../data/founta/conv/{}.tsv'.format(ds), sep='\t', index=False)

def remove_neither(ds):
  df = pd.read_csv('../data/davidson/{}.tsv'.format(ds), sep='\t')
  df = df[df.label != 2]
  print(df.label.value_counts())
  print(df.head())

  df.to_csv('../data/davidson/filtered/{}.tsv'.format(ds), sep='\t', index=False)

def oversample(ds):
  df = pd.read_csv('../data/davidson/filtered/{}.tsv'.format(ds), sep='\t')
  new_df = pd.DataFrame({'text_a':[], 'label': []})
  for i, row, in tqdm(df.iterrows(), total=df.shape[0]):
    new_df = new_df.append(row, ignore_index=True)
    if row.label == 0:
      for i in range(7):
        new_df = new_df.append(row, ignore_index=True)
  new_df.label = new_df.label.astype(int)
  new_df = new_df.sample(frac=1, random_state=42)
  new_df.to_csv('../data/davidson/filtered/{}-oversampled.tsv'.format(ds), sep='\t', index=False)


oversample('train-full')

