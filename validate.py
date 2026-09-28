"""
Валидация и подбор весов.
  python validate.py corpus — собрать корпус валидации с «трудными» соперниками → val/items.parquet
  python validate.py eval   — Recall@50 на запросах B для текущих весов и подбор весов на A (координатный спуск)
"""
import sys
import numpy as np, pandas as pd
from common import WORK_DIR, WEIGHTS, TOP_K, recall_at_k
from retriever import Retriever, FIT_USED
import dssm

D = WORK_DIR + 'val/'
N_CORPUS, K_HARD, K_LOC, POOL = 189_212, 20, 10, 500


def queries():
    return pd.read_parquet(D + 'qA.parquet'), pd.read_parquet(D + 'qB.parquet')


def build_corpus():
    """Позитивы + по K_HARD лучших по скору (без dense) + K_LOC случайных из той же локации и подкатегории
    + случайная добивка до размера benchmark. Так соперники так же плотны, как в тестовом корпусе."""
    fit, big = pd.read_parquet(D + 'fit.parquet', columns=FIT_USED), pd.read_parquet(D + 'corpus_big.parquet')
    qA, qB = queries(); q = pd.concat([qA, qB], ignore_index=True)
    R = Retriever(fit, big)
    rng = np.random.default_rng(7)
    keep = {x for r in q.rel for x in r}
    w = {k: v for k, v in WEIGHTS.items() if k != 's_dense'}
    for top in R.top_k(q, w, K_HARD):
        keep.update(big.item_id.values[top])
    by = big.groupby(['item_location_id', 'item_microcat_id']).item_id.apply(np.array)
    pos = big.set_index('item_id').loc[sorted({x for r in q.rel for x in r})]
    for key in zip(pos.item_location_id, pos.item_microcat_id):
        arr = by.get(key)
        if arr is not None:
            keep.update(rng.choice(arr, min(K_LOC, len(arr)), replace=False))
    rest = big.item_id[~big.item_id.isin(keep)].values
    keep.update(rng.choice(rest, max(0, N_CORPUS - len(keep)), replace=False))
    big[big.item_id.isin(keep)].reset_index(drop=True).to_parquet(D + 'items.parquet')
    print('val corpus', len(keep))


def pool_signals(R, q):
    """Сигналы для top-POOL кандидатов по текущим весам (для быстрого подбора весов)."""
    rows = []
    for s in range(0, len(q), 64):
        S = R.signals(q.iloc[s:s + 64]); z = R.score(S, WEIGHTS)
        top = np.argpartition(-z, POOL, axis=1)[:, :POOL]
        for i, t in enumerate(top):
            rows.append(pd.DataFrame({k: v[i, t] for k, v in S.items()}).assign(qi=s + i, item=R.corpus.item_id.values[t]))
    P = pd.concat(rows, ignore_index=True)
    rel = {(i, x) for i, r in enumerate(q.rel) for x in r}
    P['y'] = [(a, b) in rel for a, b in zip(P.qi, P.item)]
    return P


def recall(P, w, rel_n):
    z = sum(v * P[k].values for k, v in w.items() if v)
    top = P.assign(z=z).sort_values(['qi', 'z'], ascending=[True, False]).groupby('qi').head(TOP_K)
    return float((top.groupby('qi').y.sum().reindex(range(len(rel_n))).fillna(0) / rel_n).mean())


def evaluate():
    fit = pd.read_parquet(D + 'fit.parquet', columns=FIT_USED)
    R = Retriever(fit, pd.read_parquet(D + 'items.parquet'), dense=dssm.load('val'))
    qA, qB = queries()
    PA, PB = pool_signals(R, qA), pool_signals(R, qB)
    nA, nB = qA.rel.str.len().values, qB.rel.str.len().values
    ids = R.corpus.item_id.values
    for name, w in [('только текст', {'s_full': 1}), ('текст + гео', {'s_full': 1, 'loc_lp': 0.1}),
                    ('финальные веса', WEIGHTS)]:
        pred = [set(ids[t]) for t in R.top_k(qB, w, TOP_K)]
        print(f'{name}: Recall@50 B = {recall_at_k(pred, [set(r) for r in qB.rel]):.4f}', flush=True)
    # координатный спуск: по очереди меняем один вес, пока растёт Recall@50 на A
    w, best = dict(WEIGHTS), recall(PA, WEIGHTS, nA)
    for _ in range(2):
        for k in w:
            for m in [0, 0.5, 0.75, 1.33, 2]:
                w2 = dict(w, **{k: w[k] * m})
                r = recall(PA, w2, nA)
                if r > best + 1e-4:
                    best, w = r, w2
    print('подобранные веса', {k: round(v, 3) for k, v in w.items()},
          f'A = {best:.4f}, B = {recall(PB, w, nB):.4f}')


if __name__ == '__main__':
    {'corpus': build_corpus, 'eval': evaluate}[sys.argv[1]]()
