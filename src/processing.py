import pandas as pd
from tqdm import tqdm
from tokenization import Processor

tqdm.pandas()
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

def preprocess(ds):
  processor = Processor(preserve_case=False, reduce_len=False, strip_handles=False, demojize=True, replace_url=True, segment_hashtags=True, correct_user=True, url_to_http=True, remove_rt=True, change_at=False)
  path = '../data/solid/conv/{}.tsv'.format(ds)
  df = pd.read_csv(path, sep='\t')
  df.text_a = df.text_a.progress_apply(processor.process)
  df.to_csv(path, sep='\t', index=False)


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


preprocess(train)

