"""Финальный ответ: top-50 объявлений по линейному скору → answer.csv (с проверкой формата)."""
import os, time
import pandas as pd
from common import WORK_DIR, WEIGHTS, TOP_K
from retriever import Retriever, FIT_USED
import dssm

d = WORK_DIR + 'test/'
t = time.time()
R = Retriever(pd.read_parquet(d + 'fit.parquet', columns=FIT_USED), pd.read_parquet(d + 'items.parquet'),
              dense=dssm.load('test'))
q = pd.read_parquet(d + 'queries.parquet')
ids = R.corpus.item_id.values
top = R.top_k(q, WEIGHTS, TOP_K)
ans = pd.DataFrame({'query_id': q.query_id, 'answer': [' '.join(ids[i]) for i in top]})

corpus_ids = set(ids)
assert ans.query_id.is_unique and len(ans) == len(q)
for a in ans.answer:
    x = a.split()
    assert len(x) == TOP_K == len(set(x)) and all(len(i) == 16 and i in corpus_ids for i in x)
out = os.environ.get('OUT', 'answer.csv')
ans.to_csv(out, index=False)
print('saved', out, '%.0fs' % (time.time() - t))
