import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskEncoder(nn.Module):
    """Compact RoBERTa-style masked-region encoder (self-attention only,
    no RDKit, no external transformer libs). Layout mirrors
    mt_gnn_v2.GINEEncoder"""

    def __init__(self, vocab, d=128, layers=2, heads=4, ff=512, max_len=128,
                 dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, mask=None):
        x = self.embed(ids) + self.pos[:, :ids.size(1)]
        h = self.enc(x, src_key_padding_mask=mask)
        logits = self.head(h)
        return logits, h


def pool_embeddings(model, ids, device="cpu", max_len=None):
    """Mean-pool of final-layer hidden states over non-[PAD] tokens.
    Returns np.ndarray (n, d)."""
    ids = torch.as_tensor(ids, device=device)
    if max_len is not None:
        ids = ids[:, :max_len]
    model.eval()
    with torch.no_grad():
        _, h = model(ids)
    h = h.float().cpu().numpy()
    valid = (ids != 0).cpu().numpy().astype(bool)
    mask = valid[..., None]
    sums = (h * mask).sum(axis=1)
    counts = mask.sum(axis=1)[:, 0]
    counts = np.where(counts == 0, 1.0, counts)
    return (sums / counts[:, None]).astype(np.float32)


def pretrain_encoder(model, ids, epochs=2, bs=64, lr=3e-4, seed=42, mask_p=0.15):
    """Masked-token prediction. Returns list of per-batch train losses."""
    torch.manual_seed(seed)
    ids = torch.as_tensor(ids)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    mask_id = 2  # [MASK]

    losses = []
    for _ in range(epochs):
        perm = torch.randperm(ids.size(0), generator=generator)
        for start in range(0, ids.size(0), bs):
            batch = ids[perm[start:start + bs]]
            rand = torch.rand(batch.shape, generator=generator)
            to_mask = (rand < mask_p) & (batch != 0)
            targets = batch.clone()
            targets[to_mask] = -100
            masked = batch.clone()
            masked[to_mask] = mask_id
            logits, _ = model(masked)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   targets.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses