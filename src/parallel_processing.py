from mpi4py import MPI
import numpy as np
import pandas as pd
import os
from tokenization import Processor
from tqdm import tqdm

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  path = '../data/solid/conv/train.tsv'
  print("Read dataframe")
  df = pd.read_csv(path, sep='\t')
  
  text_a = df.text_a.to_numpy()
  print("Splitting arrays")

  text_a = np.array_split(text_a, size)
  print("Arrays split")
else:
  text_a = None

print("{} getting processor\n".format(rank))
processor = Processor(preserve_case=False, reduce_len=False, strip_handles=False, demojize=True, replace_url=True, segment_hashtags=True, correct_user=True, url_to_http=True, remove_rt=True, change_at=False)

text_a = comm.scatter(text_a, root=0)

print("{} got {} tweets".format(rank, len(text_a)))

text_a_tmp = []
for t in tqdm(text_a, total=len(text_a)):
  text_a_tmp.append(processor.process(t))

text_a = np.asarray(text_a_tmp)

del text_a_tmp

print("Rank {} of {} finished\n".format(rank, size))

text_a = comm.gather(text_a, root=0)

if rank == 0:
  text_a = np.concatenate(text_a)

  print(text_a.shape)
  df.text_a = text_a
  print(df.head())
  df.to_csv('../ernie/data/solid/paralell_test.tsv', index=False, sep='\t')


