import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from v22_tokenizer import (
    PROTECTED, SPECIALS, decode, encode, learn_bpe, segment, tokenize_batch,
)


def test_specials_and_pad_id():
    toks = learn_bpe(["C" * 20, "c1ccccc1", "N[*](C)C"], vocab_target=20)
    for s in SPECIALS:
        assert s in toks["tok2id"]
    assert toks["tok2id"]["[PAD]"] == 0


def test_learned_merges_never_contain_protected():
    toks = learn_bpe(["C(=O)N" * 40, "*CC(=O)O" * 40, "c1ccc(cc1)N" * 40],
                     vocab_target=80)
    for a, b in toks["merges"]:
        assert a not in PROTECTED and b not in PROTECTED


def test_protected_tokens_never_split():
    sm = "*C(C)(C)C"
    toks = learn_bpe([sm] * 200, vocab_target=60)
    ids = encode(sm, toks)
    toks_dec = [toks["id2tok"][i] for i in ids if i not in {toks["tok2id"][s] for s in SPECIALS}]
    assert "*" in toks_dec
    assert "(" in toks_dec and ")" in toks_dec


def test_round_trip():
    smiles = ["C", "CC", "c1ccccc1", "C[C@H](N)C(=O)O", "N[*](C)(C)C(=O)N"]
    toks = learn_bpe(smiles + ["C" * 50, "c1ccc(cc1)NC(=O)O" * 3], vocab_target=40)
    for s in smiles:
        ids = encode(s, toks)
        assert SPECIALS[0] not in [toks["id2tok"][i] for i in ids]
        assert decode(ids, toks) == s


def test_tokenize_batch_shape_cls_pad():
    toks = learn_bpe(["CC" * 30, "N[*](C)(C)C" * 20, "c1ccccc1"], vocab_target=30)
    x = tokenize_batch(toks, ["C" * 30, "", "c1ccccc1"], max_len=16)
    assert isinstance(x, np.ndarray) and x.shape == (3, 16)
    assert x.dtype == np.int32
    assert x[0, 0] == toks["tok2id"]["[CLS]"]
    assert (x[0, :] == 0).sum() >= 1            # padding present
    assert (x[1, :] == 0).sum() >= 14           # "" -> CLS + pad


def test_learn_bpe_deterministic():
    corpus = ["C(=O)N" * 10, "c1ccc(cc1)N" * 5, "*CC(=O)O" * 8, "CCC" * 7]
    t1 = learn_bpe(corpus, vocab_target=40)
    t2 = learn_bpe(corpus, vocab_target=40)
    assert t1["tok2id"] == t2["tok2id"]
    assert t1["merges"] == t2["merges"]


def test_stratified_subset_never_crashes():
    import numpy as np
    rng = np.random.default_rng(0)
    corpus = [("C" * (i % 12)) + ("=O" * (i % 3)) + "N" for i in range(400)]
    toks = learn_bpe(corpus, vocab_target=50, max_subset=150)
    assert len(toks["tok2id"]) >= 4
