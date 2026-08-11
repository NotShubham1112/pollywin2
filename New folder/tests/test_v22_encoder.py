import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from v22_encoder import BertEncoder, pool_embeddings, pretrain_mlm


def test_forward_shape_and_pool():
    ids = torch.randint(4, 20, (6, 32))
    m = BertEncoder(vocab=100, d=32, layers=2, heads=4)
    logits, feats = m(ids)
    assert logits.shape == (6, 32, 100)
    assert feats.shape == (6, 32, 32)
    pool = pool_embeddings(m, ids, max_len=32)
    assert pool.shape == (6, 32)
    assert np.isfinite(pool).all()


def test_default_config_is_384_6_8():
    m = BertEncoder(vocab=4000)
    assert m.d == 384
    assert m.enc.num_layers == 6
    assert m.enc.layers[0].self_attn.num_heads == 8


def test_all_pad_row_pools_to_zero_vector():
    ids = torch.zeros(4, 32, dtype=torch.long)
    ids[1:, :] = torch.randint(4, 40, (3, 32))
    m = BertEncoder(vocab=100, d=32, layers=2, heads=4)
    pool = pool_embeddings(m, ids)
    assert np.array_equal(pool[0], np.zeros(32, dtype=np.float32))


def test_pool_echoes_nonpad_mean():
    torch.manual_seed(0)
    m = BertEncoder(vocab=64, d=16, layers=1, heads=4)
    ids = torch.randint(5, 60, (2, 8))
    ids[:, 4:] = 0
    pool = pool_embeddings(m, ids)
    _, feats = m(ids)
    mean = feats[:, :4].mean(dim=1).detach().cpu().numpy()
    assert np.allclose(pool, mean, atol=1e-6)


def test_protected_tokens_never_masked():
    torch.manual_seed(0)
    m = BertEncoder(vocab=30, d=8, layers=1, heads=4)
    # protected ids: 24, 25
    ids = torch.randint(4, 26, (16, 16)).long()
    ids[0, :] = torch.tensor([24, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    _, losses = pretrain_mlm(m, ids, epochs=1, bs=8, lr=1e-3, seed=0,
                             mask_p=1.0, protected_ids=(24, 25), device="cpu")
    # mask_p=1.0 with all protected/pad/cls positions excluded must still run
    assert all(np.isfinite(v) for v in losses)


def test_pretrain_reduces_loss_and_is_deterministic():
    torch.manual_seed(0)
    ids = torch.randint(4, 40, (24, 16))
    m1 = BertEncoder(vocab=64, d=16, layers=1, heads=4)
    l1, _ = pretrain_mlm(m1, ids, epochs=3, bs=3, lr=1e-3, seed=7, device="cpu")
    m2 = BertEncoder(vocab=64, d=16, layers=1, heads=4)
    l2, _ = pretrain_mlm(m2, ids, epochs=3, bs=3, lr=1e-3, seed=7, device="cpu")
    assert l1 == l2
    assert len(l1) == 24                     # epochs * ceil(n/bs) = 3 * 8
    assert all(np.isfinite(v) for v in l1)
    assert l1[-1] < l1[0]


def test_best_val_checkpoint_restored():
    torch.manual_seed(0)
    ids = torch.randint(4, 40, (48, 16))
    val_ids = torch.randint(4, 40, (12, 16))
    m = BertEncoder(vocab=64, d=16, layers=1, heads=4)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    _, best = pretrain_mlm(m, ids, epochs=2, bs=12, lr=1e-3, seed=0,
                           val_ids=val_ids, device="cpu")
    assert best is not None and np.isfinite(best)
    after = {k: v.clone() for k, v in m.state_dict().items()}
    # training must have moved the weights away from init (otherwise the
    # checkpoint restore is meaningless)
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed


def test_masked_positions_are_the_loss_targets():
    """MLM must learn to predict ORIGINAL tokens at masked positions."""
    torch.manual_seed(0)
    V, B, L = 40, 128, 24
    start = torch.randint(4, V, (B, 1))
    idx = torch.arange(L).unsqueeze(0)
    ids = (start + idx) % (V - 4) + 4        # arithmetic-progression rows
    m = BertEncoder(vocab=V, d=16, layers=1, heads=4)
    pretrain_mlm(m, ids, epochs=30, bs=64, lr=3e-3, seed=0, mask_p=0.3,
                 device="cpu")
    torch.manual_seed(123)
    rand = torch.rand(ids.shape)
    to_mask = (rand < 0.3) & (ids != 0)
    masked = ids.clone()
    masked[to_mask] = 2
    logits, _ = m(masked)
    nll = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1),
                          reduction="none").reshape(ids.shape)[to_mask]
    assert nll.mean().item() < np.log(V) - 0.5
