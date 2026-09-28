import os, re, glob
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import snowballstemmer

DATA_DIR = os.environ.get('DATA_DIR', '.').rstrip('/') + '/'
WORK_DIR = os.environ.get('WORK_DIR', './work').rstrip('/') + '/'

# Веса линейного скора (подобраны на валидации, validate.py).
WEIGHTS = {'s_full': 2.66, 's_params': 0.399, 's_desc': 0.532, 's_mem': 0.05, 'loc_lp': 0.15, 's_dense': 1.33}
TOP_K = 50

Q_COLS = ['search_query', 'search_location_id', 'search_infm_params_text', 'search_category']
ITEM_COLS = ['item_id', 'item_title_raw', 'item_description_raw', 'item_infm_params_text',
             'item_microcat_id', 'item_location_id', 'item_latitude', 'item_longitude']

_stemmer = snowballstemmer.stemmer('russian')
_stem_cache = {}
_tok = re.compile(r'[a-zа-я0-9]+')
STOP = set('и в во на с со по для от до из к ко у о об а но или не за при под над без'.split())


def norm(s):
    """Нижний регистр, ё→е, только буквы/цифры."""
    s = str(s or '').lower().replace('ё', 'е')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s)).strip()


def stems(text):
    out = []
    for t in _tok.findall(str(text or '').lower().replace('ё', 'е')):
        if t in STOP:
            continue
        s = _stem_cache.get(t)
        if s is None:
            s = _stem_cache[t] = _stemmer.stemWord(t) if len(t) > 3 else t
        out.append(s)
    return out


def stem_str(text):
    return ' '.join(stems(text))


def read_train(columns):
    """train.parquet без тяжёлых описаний (или его разрезанная копия parts/)."""
    if os.path.exists(DATA_DIR + 'train.parquet'):
        return pq.read_table(DATA_DIR + 'train.parquet', columns=columns).to_pandas()
    return pq.read_table(DATA_DIR + 'parts/train_nodesc.parquet', columns=columns).to_pandas()


def iter_train_desc(max_len):
    """Пары (item_id, описание[:max_len]) из train потоково, чтобы не держать все описания в памяти."""
    files = ([DATA_DIR + 'train.parquet'] if os.path.exists(DATA_DIR + 'train.parquet')
             else sorted(glob.glob(DATA_DIR + 'parts/desc_*.parquet')))
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(100_000, columns=['item_id', 'item_description_raw']):
            yield (b.column(0).to_numpy(zero_copy_only=False),
                   pc.utf8_slice_codeunits(pc.fill_null(b.column(1), ''), 0, max_len).to_numpy(zero_copy_only=False))


def recall_at_k(pred, rel):
    """pred, rel — списки множеств item_id по запросам."""
    return float(np.mean([len(p & r) / len(r) for p, r in zip(pred, rel)]))
