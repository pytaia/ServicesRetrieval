"""
Подготовка данных.
  python prepare.py test — train целиком + benchmark → WORK_DIR/test/
  python prepare.py val  — разбиение train на «два мира» для валидации → WORK_DIR/val/
"""
import os, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from common import DATA_DIR, WORK_DIR, Q_COLS, norm, read_train, iter_train_desc

FIT_COLS = ['search_query', 'search_location_id', 'search_infm_params_text',
            'item_id', 'item_title_raw', 'item_infm_params_text', 'item_location_id', 'item_microcat_id']
CORPUS_COLS = ['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw', 'item_location_id']
SEED, N_HOLD, N_C = 42, 3000, 200_000


def attach_desc(items, max_len):
    """Добавить описания train-объявлений (обрезанные) к таблице объявлений."""
    need = set(items.item_id);
    parts = []
    for ids, desc in iter_train_desc(max_len):
        m = np.fromiter((x in need for x in ids), bool, len(ids))
        parts.append(pd.Series(desc[m], index=ids[m]))
    d = pd.concat(parts);
    d = d[~d.index.duplicated()]
    return items.assign(item_description_raw=d.reindex(items.item_id).fillna('').values)


def save(out, **tables):
    os.makedirs(out, exist_ok=True)
    for name, t in tables.items():
        t.reset_index(drop=True).to_parquet(out + name + '.parquet')


def train_items(fit):
    """Уникальные объявления fit с описаниями — для обучения bi-encoder."""
    it = fit.drop_duplicates('item_id')[['item_id', 'item_title_raw', 'item_infm_params_text']]
    return attach_desc(it, 300)


def prepare_test():
    fit = read_train(FIT_COLS)
    items = pq.read_table(DATA_DIR + 'benchmark_items.parquet', columns=CORPUS_COLS).to_pandas()
    items['item_description_raw'] = items.item_description_raw.fillna('').str.slice(0, 700)
    queries = pd.read_parquet(DATA_DIR + 'benchmark_queries.parquet', columns=['query_id'] + Q_COLS)
    save(WORK_DIR + 'test/', fit=fit, items=items, queries=queries, train_items=train_items(fit))


def prepare_val():
    """
    Валидация «двух миров»: N_C случайных train-объявлений становятся «чужими» (корпус),
    fit — строки train с остальными объявлениями. Запросы A/B — поиски, все выбранные объявления
    которых чужие. 62.5% из них — редкие тексты, целиком убранные из fit (в benchmark 37.5% текстов знакомы).
    """
    tr = read_train(FIT_COLS)
    tr['qn'] = tr.search_query.map(norm)
    tr['g'] = tr.qn + '|' + tr.search_location_id.astype(str) + '|' + tr.search_infm_params_text.fillna('').map(norm)
    rng = np.random.default_rng(SEED)
    C = pd.Index(rng.choice(tr.item_id.unique(), N_C, replace=False))
    tr['inC'] = tr.item_id.isin(C)
    elig = tr.groupby('g').agg(ok=('inC', 'min'), qn=('qn', 'first'))
    elig = elig[elig.ok]
    n_tot = 2 * N_HOLD;
    n_new = int(0.625 * n_tot)
    qn_cnt = tr.qn.value_counts()
    rare = elig[elig.qn.map(qn_cnt) <= 10]
    g_new = rng.choice(rare.index.values, n_new, replace=False)
    q_new = set(elig.loc[g_new, 'qn'])
    fit_mask = ~tr.inC & ~tr.qn.isin(q_new)
    fit_qn = set(tr.loc[fit_mask, 'qn'])
    old = elig[elig.qn.isin(fit_qn)].index.values
    g_old = rng.choice(old, n_tot - n_new, replace=False)
    sel = np.concatenate([g_new, g_old]);
    rng.shuffle(sel)

    def queries(gs, name):
        q = (tr[tr.g.isin(set(gs))].groupby('g')
             .agg(**{c: (c, 'first') for c in Q_COLS if c in tr}, rel=('item_id', lambda s: sorted(set(s))))
             .reset_index(drop=True))
        q['query_id'] = [f'{name}{i:05d}' for i in range(len(q))]
        return q

    fit = tr.loc[fit_mask, FIT_COLS]
    big = tr[tr.inC].drop_duplicates('item_id')[['item_id', 'item_title_raw', 'item_infm_params_text',
                                                 'item_location_id', 'item_microcat_id']]
    big = attach_desc(big, 700)
    save(WORK_DIR + 'val/', fit=fit, corpus_big=big, qA=queries(sel[:N_HOLD], 'A'),
         qB=queries(sel[N_HOLD:], 'B'), train_items=train_items(fit))
    print('fit', len(fit), 'corpus_big', len(big))


if __name__ == '__main__':
    {'test': prepare_test, 'val': prepare_val}[sys.argv[1]]()
