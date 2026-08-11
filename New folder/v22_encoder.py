import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 2


class BertEncoder(nn.Module):
    """In-notebook BERT-style MLM encoder (pure torch, no external libs)."""

    def __init__(self, vocab, d=384, layers=6, heads=8, ff=None, max_len=128,
                 dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.max_len = max_len
        ff = ff or 4 * d
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, vocab)

    def reset_parameters(self):
        """Deterministically re-initialize all weights from the current RNG seed."""
        self.embed.reset_parameters()
        nn.init.normal_(self.pos, std=0.02)
        for layer in self.enc.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.constant_(layer.self_attn.out_proj.bias, 0.0)
            layer.linear1.reset_parameters()
            layer.linear2.reset_parameters()
            layer.norm1.reset_parameters()
            layer.norm2.reset_parameters()
        self.head.reset_parameters()

    def forward(self, ids, mask=None):
        x = self.embed(ids) + self.pos[:, :ids.size(1)]
        h = self.enc(x, src_key_padding_mask=mask)
        logits = self.head(h)
        return logits, h


def pool_embeddings(model, ids, max_len=None):
    """Mean-pool of final-layer hidden states over non-[PAD] tokens -> (n, d)."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    device = next(model.parameters()).device
    ids = ids.to(device)
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


def _eval_nll(model, ids, mask_p, mask_id, protected_ids, bs):
    model.eval()
    total, cnt = 0.0, 0
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=next(model.parameters()).device)
    for start in range(0, ids_t.size(0), bs):
        batch = ids_t[start:start + bs]
        torch.manual_seed(0)
        rand = torch.rand(batch.cpu().shape)
        maskable = (batch != 0) & (batch != 1) & (batch != mask_id)
        for pid in protected_ids:
            maskable = maskable & (batch != pid)
        to_mask = (rand < mask_p).to(batch.device) & maskable
        if not to_mask.any():
            continue
        targets = torch.full_like(batch, -100)
        targets[to_mask] = batch[to_mask]
        masked = batch.clone()
        masked[to_mask] = mask_id
        logits, _ = model(masked)
        nll = F.cross_entropy(logits.reshape(-1, model.vocab),
                              targets.reshape(-1), reduction="none")
        keep = (targets.reshape(-1) != -100)
        total += float(nll[keep].sum().item())
        cnt += int(keep.sum().item())
    model.train()
    return total / max(1, cnt)


def pretrain_mlm(model, ids, epochs=1, bs=256, lr=3e-4, seed=42, mask_p=0.15,
                 mask_id=MASK_ID, protected_ids=(), val_ids=None, device="cpu"):
    """Masked-token prediction with cosine LR + best-val checkpoint restore.

    Returns (losses, best_val_nll). Protected and special tokens are never
    masked. When val_ids is provided, the model is restored to the state with
    the lowest val NLL (deterministic eval mask). When val_ids is None the
    second return value is the losses list itself.
    """
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    model.to(device)
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
    n = ids_t.size(0)
    opt = torch.optim.AdamW(model.parameters(), lr=5.0 * lr)
    steps_per_epoch = max(1, int(np.ceil(n / bs)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=2 * epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(seed)

    losses = []
    best_state, best_nll = None, float("inf")
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, bs):
            batch = ids_t[perm[start:start + bs]]
            rand = torch.rand(batch.shape, generator=gen)
            maskable = (batch != 0) & (batch != 1) & (batch != mask_id)
            for pid in protected_ids:
                maskable = maskable & (batch != pid)
            to_mask = (rand < mask_p) & maskable
            if not to_mask.any():
                losses.append(0.0)
                continue
            targets = torch.full_like(batch, -100)
            targets[to_mask] = batch[to_mask]
            masked = batch.clone()
            masked[to_mask] = mask_id
            logits, _ = model(masked)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   targets.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if val_ids is not None:
            vn = _eval_nll(model, val_ids, mask_p, mask_id, protected_ids, bs)
            if vn < best_nll:
                best_nll = vn
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.random.set_rng_state(rng_state)
    return losses, (best_nll if best_state is not None else losses)
