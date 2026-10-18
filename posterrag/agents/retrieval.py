"""RetrievalAgents 1-3: categorical filtering -> narrative matching -> aesthetic selection over the RAP-25K pool."""
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor

from ..llm import extract_json
from .parser import render_prompt

RELAXATION = (('domain', 'task', 'method'), ('domain', 'task'), ('domain', 'method'), ('domain',))
DIMS = ('content_density', 'visualization_ratio', 'scientific_aesthetics')


def categorical_filter(pool, labels, min_candidates=5):
    """RetrievalAgent-1: templates whose (domain, task, method) match; relaxed step by step when too few remain."""
    for keys in RELAXATION:
        cands = [i for i, r in enumerate(pool.rows) if all(r.get(k) == labels.get(k) for k in keys)]
        if len(cands) >= min_candidates:
            return cands, '+'.join(keys)
    return list(range(len(pool.rows))), 'all'


def _tfidf_prefilter(query, cands, pool, top_k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    valid = [i for i in cands if pool.rows[i].get('intro_summary')]
    if not query or len(valid) <= top_k: return valid[:top_k] or cands[:top_k]
    X = TfidfVectorizer(max_features=5000, stop_words='english').fit_transform([query] + [pool.rows[i]['intro_summary'] for i in valid])
    sims = cosine_similarity(X[0:1], X[1:]).ravel()
    return [valid[j] for j in sims.argsort()[::-1][:top_k]]


def narrative_match(pool, cands, intro_summary, llm, tfidf_top_k=15, top_k=3):
    """RetrievalAgent-2: rank candidates by narrative similarity of introduction summaries (TF-IDF shortlist, LLM re-rank)."""
    short = _tfidf_prefilter(intro_summary, cands, pool, tfidf_top_k)
    if len(short) <= top_k or not intro_summary: return short[:top_k]
    system, prompt = render_prompt('narrative_rerank', query=intro_summary[:1500], candidates=[pool.rows[i]['intro_summary'][:700] for i in short], top_k=top_k)
    r = llm.text(prompt, system=system, temperature=0.1, max_tokens=256)
    m = re.search(r'\[[\d\s,]+\]', r['content'])
    if m:
        order = [short[int(k) - 1] for k in json.loads(m.group()) if 0 < int(k) <= len(short)]
        if order: return order[:top_k]
    return short[:top_k]


def preference_weights(preferences, llm):
    """VLM/LLM intent recognizer: free-text style preferences -> per-dimension target and weight."""
    system, prompt = render_prompt('preference_weights', preferences=preferences)
    ans = extract_json(llm.text(prompt, system=system, temperature=0.1, max_tokens=512)['content'])
    return {d: {'target': float(ans.get(d, {}).get('target', 3)), 'weight': float(ans.get(d, {}).get('weight', 1))} for d in DIMS}


def score_poster(path, llm):
    system, prompt = render_prompt('aesthetic_score')
    ans = extract_json(llm.vision(prompt, [path], system=system, temperature=0.1, max_tokens=256)['content'])
    return {d: int(ans[d]) for d in DIMS}


def aesthetic_select(pool, cands, llm, preferences=None, workers=3):
    """RetrievalAgent-3: score each candidate poster on three 1-5 scales and pick the one nearest the (weighted) target;
    the default target is the neutral midpoint (3, 3, 3) with equal weights."""
    weights = preference_weights(preferences, llm) if preferences else {d: {'target': 3.0, 'weight': 1.0} for d in DIMS}
    def one(i):
        try: return i, score_poster(pool.fetch_poster(pool.rows[i]), llm)
        except Exception as e:                                                                       # noqa: BLE001
            print(f'  [RetrievalAgent-3] {pool.rows[i]["paper_id"]}: {e}'); return i, None
    with ThreadPoolExecutor(workers) as ex: scored = [s for s in ex.map(one, cands) if s[1]]
    if not scored: return cands[0], {}, weights
    def dist(s): return math.sqrt(sum(weights[d]['weight'] * (s[d] - weights[d]['target']) ** 2 for d in DIMS))
    scored.sort(key=lambda t: dist(t[1]))
    return scored[0][0], {pool.rows[i]['paper_id']: s for i, s in scored}, weights


def retrieve(pool, pmeta, llm, preferences=None, exclude_ids=()):
    cands, level = categorical_filter(pool, pmeta['labels'])
    cands = [i for i in cands if pool.rows[i]['paper_id'] not in set(exclude_ids)]
    print(f'[RetrievalAgent-1] {len(cands)} templates match on {level}')
    top = narrative_match(pool, cands, pmeta['intro_summary'], llm)
    print(f'[RetrievalAgent-2] shortlist: {[pool.rows[i]["paper_id"] for i in top]}')
    win, scores, weights = aesthetic_select(pool, top, llm, preferences)
    print(f'[RetrievalAgent-3] selected {pool.rows[win]["paper_id"]} {scores.get(pool.rows[win]["paper_id"])}')
    return {'template': pool.rows[win], 'level': level, 'n_candidates': len(cands), 'shortlist': [pool.rows[i]['paper_id'] for i in top],
            'aesthetic_scores': scores, 'weights': weights}
