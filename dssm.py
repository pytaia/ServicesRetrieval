"""
Bi-encoder (две «башни»: для запросов и для объявлений), обучается с нуля на парах «запрос — выбранное объявление».
Текст → мешок хешированных признаков (стеммы, биграммы, символьные 3-граммы) → EmbeddingBag → вектор.
Лосс InfoNCE с in-batch негативами.
  python dssm.py test|val — обучить на WORK_DIR/<mode>/fit.parquet и посчитать эмбеддинги items.parquet
"""
import sys, time, zlib, gc
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from common import WORK_DIR, stems

NB, DIM = 1 << 18, 192          # размер хеш-пространства и эмбеддинга
EPOCHS, BS, LR = 3, 1024, 3e-3
torch.set_num_threads(2)
_fc = {}


def _h(s):
    return zlib.crc32(s.encode()) & (NB - 1)


def feats(text, p):
    """'сильная часть | слабая часть': в сильной (запрос/заголовок) ещё и символьные 3-граммы."""
    strong, _, weak = str(text).partition(' | ')
    out, st = set(), stems(strong)
    for w in st:
        f = _fc.get(p + w)
        if f is None:
            ww = '#' + w + '#'
            f = _fc[p + w] = [_h(p + 'w' + w)] + [_h(p + 'c' + ww[i:i + 3]) for i in range(len(ww) - 2)]
        out.update(f)
    out.update(_h(p + 'b' + a + '_' + b) for a, b in zip(st, st[1:]))
    sw = stems(weak)
    out.update(_h(p + 'x' + w) for w in sw)
    out.update(_h(p + 'y' + a + '_' + b) for a, b in zip(sw, sw[1:]))
    return list(out) or [0]


def _merge(parts):
    lens = np.concatenate([p[1] for p in parts])
    return np.concatenate([p[0] for p in parts]), lens, np.concatenate([[0], np.cumsum(lens)[:-1]])


def encode(texts, p):
    """Тексты → (плоский массив id признаков, длины, смещения)."""
    arrs = [np.asarray(feats(t, p), np.int32) for t in texts]
    return _merge([(np.concatenate(arrs), np.array([len(a) for a in arrs], np.int64))])


def encode_items(df, chunk=50_000):
    """Объявления кодируются частями, чтобы не держать в памяти все тексты сразу."""
    return _merge([encode(i_texts(df.iloc[s:s + chunk]), 'd') for s in range(0, len(df), chunk)])


def q_texts(df):
    return (df.search_query.fillna('') + ' | ' + df.search_infm_params_text.fillna('')).tolist()


def i_texts(df):
    return (df.item_title_raw.fillna('') + ' | ' + df.item_infm_params_text.fillna('') + ' ' +
            df.item_description_raw.fillna('').str.slice(0, 300)).tolist()


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(NB, DIM, mode='sum', sparse=True)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, ids, off, lens):
        x = self.emb(ids, off) / torch.sqrt(lens.float()).unsqueeze(1)
        return F.normalize(self.lin(torch.tanh(x)) + x, dim=-1)


class BiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.d = Tower(), Tower()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.05), dtype=torch.float32))


def _batch(enc, idx):
    flat, lens, st = enc
    l = lens[idx]
    ids = np.concatenate([flat[a:a + b] for a, b in zip(st[idx], l)]).astype(np.int64)
    return torch.from_numpy(ids), torch.from_numpy(np.concatenate([[0], np.cumsum(l)[:-1]])), torch.from_numpy(l)


@torch.no_grad()
def embed(tower, data, p, bs=4096):
    """data — список текстов запросов (p="q") или таблица объявлений (p="d")."""
    enc = encode(data, p) if p == "q" else encode_items(data)
    return np.concatenate([tower(*_batch(enc, np.arange(s, min(s + bs, len(data))))).numpy()
                           for s in range(0, len(data), bs)]).astype(np.float32)


def train(qenc, denc, qidx, didx):
    torch.manual_seed(0)
    m = BiEncoder()
    opt_s = torch.optim.SparseAdam([m.q.emb.weight, m.d.emb.weight], lr=LR)
    opt_d = torch.optim.Adam([p for n, p in m.named_parameters() if 'emb' not in n], lr=LR)
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        perm, tot = rng.permutation(len(qidx)), []
        for s in range(0, len(perm) - BS + 1, BS):
            b = perm[s:s + BS]
            eq, ed = m.q(*_batch(qenc, qidx[b])), m.d(*_batch(denc, didx[b]))
            logits = m.logit_scale.exp().clamp(max=100) * eq @ ed.T
            k = torch.from_numpy(didx[b])
            logits = logits.masked_fill((k[:, None] == k[None]) & ~torch.eye(len(b), dtype=torch.bool), -1e4)  # дубли
            lab = torch.arange(len(b))
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt_s.zero_grad(); opt_d.zero_grad(); loss.backward(); opt_s.step(); opt_d.step()
            tot.append(loss.item())
        print(f'epoch {ep} loss {np.mean(tot):.4f}', flush=True)
    return m.eval()


def main(mode):
    d = WORK_DIR + mode + '/'
    t = time.time()
    fit = pd.read_parquet(d + 'fit.parquet', columns=['search_query', 'search_infm_params_text', 'item_id'])
    ti = pd.read_parquet(d + 'train_items.parquet').set_index('item_id')
    qidx, qu = pd.factorize(pd.Series(q_texts(fit)))
    didx = ti.index.get_indexer(fit.item_id)
    del fit
    qenc = encode(list(qu), 'q')
    denc = encode_items(ti)
    del qu, ti; _fc.clear(); gc.collect()
    m = train(qenc, denc, qidx, didx)
    del qenc, denc; gc.collect()
    items = pd.read_parquet(d + 'items.parquet', columns=['item_title_raw', 'item_infm_params_text', 'item_description_raw'])
    np.save(d + 'emb.npy', embed(m.d, items, 'd'))
    torch.save(m.state_dict(), d + 'dssm.pt')
    print('dssm done %.0fs' % (time.time() - t))


def load(mode):
    m = BiEncoder(); m.load_state_dict(torch.load(WORK_DIR + mode + '/dssm.pt')); m.eval()
    return m.q, np.load(WORK_DIR + mode + '/emb.npy')


if __name__ == '__main__':
    main(sys.argv[1])
