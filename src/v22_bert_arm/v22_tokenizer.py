import re
from collections import Counter

import numpy as np

SPECIALS = ("[PAD]", "[CLS]", "[MASK]", "[UNK]")
PROTECTED = ("*", "(", ")")
_BRACKET = re.compile(r"(\[[^\]]+\])")


def segment(smiles, protect=PROTECTED):
    """Initial atomic token units. Protected tokens are always their own unit."""
    out = []
    i = 0
    s = smiles
    while i < len(s):
        if s[i] == "[":
            m = _BRACKET.match(s, i)
            if m:
                out.append(m.group(1))
                i = m.end()
                continue
        two = s[i:i + 2]
        if two in ("Br", "Cl", "Si"):
            out.append(two)
            i += 2
            continue
        ch = s[i]
        if ch in protect:
            out.append(ch)          # never merged with anything
        else:
            out.append(ch)
        i += 1
    return out


def _stratified_subset(smiles, max_subset, seed):
    if len(smiles) <= max_subset:
        return list(smiles)
    rng = np.random.default_rng(seed)
    lens = np.array([len(s) for s in smiles])
    buckets = np.digitize(lens, np.quantile(lens, [0.25, 0.5, 0.75]))
    per = max(1, int(np.ceil(max_subset / len(np.unique(buckets)))))
    chosen = []
    for b in np.unique(buckets):
        idx = np.where(buckets == b)[0]
        take = min(per, idx.size)
        chosen.extend(rng.choice(idx, take, replace=False).tolist())
    # trim if over budget
    rng.shuffle(chosen)
    return [smiles[k] for k in chosen[:max_subset]]


def _apply_merge(tokens, a, b):
    nt = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            nt.append(a + b)
            i += 2
        else:
            nt.append(tokens[i])
            i += 1
    return nt


def learn_bpe(smiles, vocab_target=4000, protect=PROTECTED, seed=42,
              max_subset=150000):
    corpus = [_stratified_subset(smiles, max_subset, seed)]
    pieces = [segment(s) for s in corpus[0]]
    merges = []
    vocab = list(SPECIALS)
    seen = set(SPECIALS)
    for toks in pieces:
        for t in toks:
            if t not in seen:
                seen.add(t)
                vocab.append(t)
    while len(merges) < vocab_target:
        pairs = Counter()
        for toks in pieces:
            for a, b in zip(toks, toks[1:]):
                if a in protect or b in protect:
                    continue
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        pieces = [_apply_merge(toks, a, b) for toks in pieces]
        merged = a + b
        merges.append((a, b))
        if merged not in seen:
            seen.add(merged)
            vocab.append(merged)
    tok2id = {t: i for i, t in enumerate(vocab)}
    id2tok = {i: t for i, t in enumerate(vocab)}
    return {"tok2id": tok2id, "id2tok": id2tok, "merges": merges,
            "protect": tuple(protect)}


def encode(smiles, tok):
    toks = segment(smiles, tok["protect"])
    for a, b in tok["merges"]:
        toks = _apply_merge(toks, a, b)
    unk = tok["tok2id"]["[UNK]"]
    return [tok["tok2id"].get(t, unk) for t in toks]


def decode(ids, tok):
    special_ids = {tok["tok2id"][s] for s in SPECIALS}
    return "".join(tok["id2tok"][i] for i in ids if i not in special_ids)


def tokenize_batch(tok, smiles, max_len=128):
    cls, pad = tok["tok2id"]["[CLS]"], tok["tok2id"]["[PAD]"]
    out = []
    for sm in smiles:
        row = [cls] + encode(sm, tok)
        if len(row) > max_len:
            row = row[:max_len]
        row = row + [pad] * (max_len - len(row))
        out.append(row)
    return np.asarray(out, dtype=np.int32)
