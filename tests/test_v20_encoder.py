import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from v20_encoder import MaskEncoder, pool_embeddings, pretrain_encoder


def test_forward_shape_and_pool():
    ids = torch.randint(4, 20, (6, 32))
    m = MaskEncoder(vocab=100)
    _, feats = m(ids)
    assert feats.shape == (6, 32, 128)
    pool = pool_embeddings(m, ids, max_len=32)
    assert pool.shape == (6, 128)
    assert np.isfinite(pool).all()


def test_logits_shape_and_finite():
    ids = torch.randint(4, 80, (2, 20))
    m = MaskEncoder(vocab=100)
    logits, feats = m(ids)
    assert logits.shape == (2, 20, 100)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(feats).all()


def test_all_pad_row_pools_to_zero_vector():
    ids = torch.zeros(4, 32, dtype=torch.long)
    ids[1:, :] = torch.randint(4, 40, (3, 32))
    m = MaskEncoder(vocab=100)
    pool = pool_embeddings(m, ids)
    assert pool.shape == (4, 128)
    assert np.isfinite(pool).all()
    assert np.array_equal(pool[0], np.zeros(128, dtype=np.float32))


def test_pool_echoes_nonpad_mean():
    torch.manual_seed(0)
    m = MaskEncoder(vocab=64)
    ids = torch.randint(5, 60, (2, 8))
    ids[:, 4:] = 0
    pool = pool_embeddings(m, ids)
    _, feats = m(ids)
    mean = feats[:, :4].mean(dim=1).detach().cpu().numpy()
    assert np.allclose(pool, mean, atol=1e-6)


def test_masked_positions_are_the_loss_targets():
    """Regression: pretrain_encoder must learn to predict ORIGINAL tokens at
    masked positions (MLM), not copy the visible unmasked tokens.

    ids are structured (arithmetic progressions) so masked tokens ARE
    predictable from context once the model is trained on them. After
    pretraining, mean cross-entropy at masked positions (targets = original
    tokens) must fall well below uniform guessing (ln V); the buggy loss
    construction (masked positions -> -100) never trains them and stays at or
    above uniform.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    V, B, L = 40, 128, 24
    start = torch.randint(4, V, (B, 1))
    idx = torch.arange(L).unsqueeze(0)
    ids = (start + idx) % (V - 4) + 4  # each row = arithmetic progression

    torch.manual_seed(0)
    m = MaskEncoder(vocab=V)
    pretrain_encoder(m, ids, epochs=30, bs=64, lr=3e-3, seed=0, mask_p=0.3)

    torch.manual_seed(123)  # fresh, deterministic eval mask
    rand = torch.rand(ids.shape)
    to_mask = (rand < 0.3) & (ids != 0)
    masked = ids.clone()
    masked[to_mask] = 2
    logits, _ = m(masked)
    nll = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1),
                          reduction="none").reshape(ids.shape)[to_mask]
    masked_nll = nll.mean().item()
    assert masked_nll < np.log(V) - 0.5, (
        f"masked-position NLL {masked_nll:.3f} not below uniform-ish bound "
        f"{np.log(V) - 0.5:.3f}; masked positions are not being learned")


def test_pretrain_deterministic_and_reduces_loss():
    torch.manual_seed(0)
    ids = torch.randint(4, 40, (24, 16))
    V = 64

    def fresh():
        torch.manual_seed(0)
        return MaskEncoder(vocab=V)

    m1 = fresh()
    l1 = pretrain_encoder(m1, ids, epochs=3, bs=3, lr=1e-3, seed=7)
    m2 = fresh()
    l2 = pretrain_encoder(m2, ids, epochs=3, bs=3, lr=1e-3, seed=7)
    assert l1 == l2
    assert len(l1) == 24  # epochs * ceil(n / bs) = 3 * 8
    assert all(np.isfinite(v) for v in l1)
    assert l1[-1] < l1[0]