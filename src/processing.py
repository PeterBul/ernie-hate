import pandas as pd

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

remove_neither('train')
