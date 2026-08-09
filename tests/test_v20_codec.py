import numpy as np, pytest
from v20_codec import build_tokenizer, tokenize_batch

def test_vocab_specials_and_idx():
    toks = build_tokenizer(["C"*5, "c1ccccc1", "[Fe]"])
    for s in ("[PAD]", "[CLS]", "[MASK]", "[UNK]"):
        assert s in toks["tok2id"]
    assert toks["tok2id"]["[PAD]"] == 0

def test_tokenize_properties():
    toks = build_tokenizer(["CCCC"*10, "N[Fe]Cl"*3])
    x = tokenize_batch(toks, ["CCCC"*10, "", "N[Fe]Cl"*3], max_len=16)
    assert isinstance(x, np.ndarray) and x.shape == (3, 16)
    assert x[0,0] == toks["tok2id"]["[CLS]"]
    assert (x[1,:] == 0).sum() >= 3       # padding present ("" encodes to 1 token + 15 pads)
    assert x.dtype == np.int32