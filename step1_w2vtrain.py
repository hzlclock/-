#%%
import sqlite3
import pkuseg

con=sqlite3.connect('dulce.db')
seg=pkuseg.pkuseg(model_name='web')
#%%
from gensim.utils import save_as_line_sentence
import tqdm

sentenses=[]

for item in con.execute('select content from 非主流情话'):
    sentenses.append(seg.cut(item[0]))

save_as_line_sentence(tqdm.tqdm(sentenses), 'corpus')
# %%
from gensim.models import Word2Vec
word2vec_model=Word2Vec(corpus_file='corpus', size=100, workers=8, min_count=1, iter=20)
word2vec_model.save('model')

# %%
import numpy as np

words_wp = []
embeddings_wp = []
for word in list(word2vec_model.wv.vocab):
    embeddings_wp.append(word2vec_model.wv[word])
    words_wp.append(word)
with open('step1_0909.tsv', 'w') as f:
    for item in embeddings_wp:
        f.write(np.array2string(item, separator='\t').replace('[','').replace(']','').replace('\n','\t')+'\n')
# with open('step2_0909.tsv', 'w') as f:
#     f.write('\n'.join(words_wp))

# %%

import io
with io.open('step2_0909.tsv', "w", encoding="utf-8") as f:
    for i in words_wp:
        f.write(i+'.\n')
    # f.write('\n'.join(words_wp))

# %%
