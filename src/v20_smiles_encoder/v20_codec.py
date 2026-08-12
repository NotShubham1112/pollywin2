import re
from collections import Counter

import numpy as np

_SPECIALS = ("[PAD]", "[CLS]", "[MASK]", "[UNK]")
_TOK = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|\*|[A-Z][a-z]?|[0-9]{2}|[0-9]|[()\[\]=#\\/@+%.])")


def build_tokenizer(smiles, max_vocab=1600, min_count=2):
    counter = Counter()
    for sm in smiles:
        counter.update(_TOK.findall(sm))
    freq = [t for t, c in counter.items() if c >= min_count]
    freq.sort(key=lambda t: (-counter[t], t))
    cap = max(0, max_vocab - len(_SPECIALS))
    freq = freq[:cap]
    tok2id = {s: i for i, s in enumerate(_SPECIALS)}
    for t in freq:
        tok2id[t] = len(tok2id)
    id2tok = {i: t for t, i in tok2id.items()}
    return {"tok2id": tok2id, "id2tok": id2tok}


def tokenize_batch(tok, smiles, max_len=128):
    t2i = tok["tok2id"]
    cls, unk = t2i["[CLS]"], t2i["[UNK]"]
    ids = [[cls] + [t2i.get(t, unk) for t in _TOK.findall(sm)] for sm in smiles]
    for row in ids:
        if len(row) > max_len:
            del row[max_len:]
        row.extend([0] * (max_len - len(row)))
    return np.asarray(ids, dtype=np.int32)