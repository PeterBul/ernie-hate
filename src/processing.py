import pandas as pd

mapping = {0:1, 1:0, 2:2, 3:2}

df = pd.read_csv('../data/founta/test.tsv', sep='\t')

df.label = df.label.apply(lambda l: mapping[l])

print(df.head())

df.to_csv('../data/founta/test-conv.tsv', sep='\t')