import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from common import norm, stem_str
import dssm

FIT_USED = ['search_query', 'search_location_id', 'item_title_raw', 'item_location_id']


def _ngrams(s):
    t = s.split()
    return t + [a + '_' + b for a, b in zip(t, t[1:])]


class Retriever:
    def __init__(self, fit, corpus, dense=None):
        self.corpus = corpus.reset_index(drop=True)
        c = self.corpus
        title, par = c.item_title_raw.map(stem_str), c.item_infm_params_text.map(stem_str)
        desc = c.item_description_raw.fillna('').str.slice(0, 700).map(stem_str)
        self.vw = TfidfVectorizer(analyzer=_ngrams, min_df=3, max_features=400_000, sublinear_tf=True, dtype=np.float32)
        self.M_full = self.vw.fit_transform(title + ' ' + title + ' ' + par + ' ' + desc)
        self.M_par, self.M_desc = self.vw.transform(par), self.vw.transform(desc)
        del title, par, desc

        # логовая память: нормализованный заголовок → склейка train-запросов
        mem = (fit.assign(t=fit.item_title_raw.map(norm), q=fit.search_query.map(stem_str))
               .groupby('t').q.agg(' '.join))
        self.M_mem = self.vw.transform(c.item_title_raw.map(norm).map(mem).fillna(''))

        # гео-приор
        self.iloc_codes, locs = pd.factorize(c.item_location_id)
        pos = {l: i for i, l in enumerate(locs)}
        n = fit.groupby(['search_location_id', 'item_location_id']).size()
        self.loc_lp = {}
        for sl, sub in n.groupby(level=0):
            v = np.zeros(len(locs), np.float32)
            for (_, l), k in sub.items():
                if l in pos:
                    v[pos[l]] = k
            if sl in pos:
                v[pos[sl]] += 2.0
            self.loc_lp[sl] = np.log((v + 0.01) / (sub.sum() + 2.0 + 0.01 * len(v))).astype(np.float32)
        self.loc_default = np.full(len(locs), -np.log(len(locs)), np.float32)
        self.dense = dense  # (башня запросов, эмбеддинги корпуса) или None

    def signals(self, q):
        qs = q.search_query.map(stem_str)
        Qw = self.vw.transform(qs)
        Qp = self.vw.transform(qs + ' ' + q.search_infm_params_text.fillna('').map(stem_str))
        dot = lambda A, B: (A @ B.T).toarray()
        S = dict(s_full=dot(Qw, self.M_full), s_desc=dot(Qw, self.M_desc),
                 s_params=dot(Qp, self.M_par), s_mem=dot(Qw, self.M_mem),
                 loc_lp=np.stack([self.loc_lp.get(s, self.loc_default)[self.iloc_codes] for s in q.search_location_id]))
        if self.dense is not None:
            tower, E = self.dense
            S['s_dense'] = dssm.embed(tower, dssm.q_texts(q), 'q') @ E.T
        return S

    @staticmethod
    def score(S, w):
        return sum(v * S[k] for k, v in w.items() if v)

    def top_k(self, q, w, k, batch=64):
        """Индексы top-k объявлений корпуса для каждого запроса."""
        out = []
        for s in range(0, len(q), batch):
            z = self.score(self.signals(q.iloc[s:s + batch]), w)
            top = np.argpartition(-z, k, axis=1)[:, :k]
            out.extend(t[np.argsort(-z[i, t])] for i, t in enumerate(top))
        return out
