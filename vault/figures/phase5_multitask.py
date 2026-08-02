import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join("vault", "figures")

bench = pd.read_csv(os.path.join(OUT, "model_benchmark.csv"))
fig, ax = plt.subplots(figsize=(8, 4.5))
piv = bench.pivot(index="model", columns="target", values="rmse")
piv.plot(kind="bar", ax=ax, color=["#2a6fb0", "#d1495b", "#4c9a2a"])
ax.set_title("5-fold CV RMSE by model (RDKit desc + Morgan2048 + MACCS + polymer feats)")
ax.set_ylabel("RMSE")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_benchmark.png")); plt.close(fig)
print("saved model_benchmark.png")

# Phase 5: cross-target correlation using molecules labeled with >1 target
train = pd.read_csv("official_dataset/train.csv")
train["canon"] = train["smiles"].str.replace("*", "", regex=False).str.replace(r"\[\*\]", "", regex=True)
shares = train.groupby("canon")["target_type"].apply(list)
multi = shares[shares.apply(len) > 1]
print(f"\nMolecules with >1 target label: {len(multi)}")

# Build pairwise target correlation via shared molecules (median of paired labels)
pairs = []
for canon, types in multi.items():
    sub = train[train["canon"] == canon]
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            t1, t2 = sub.iloc[i]["target_type"], sub.iloc[j]["target_type"]
            if t1 != t2:
                pairs.append((t1, t2, sub.iloc[i]["target"], sub.iloc[j]["target"]))
pair_df = pd.DataFrame(pairs, columns=["t1", "t2", "y1", "y2"])
print(f"Cross-target label pairs: {len(pair_df)}")

targets = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]
corr_mat = pd.DataFrame(index=targets, columns=targets, dtype=float)
count_mat = pd.DataFrame(0, index=targets, columns=targets, dtype=int)
for (t1, t2), grp in pair_df.groupby(["t1", "t2"]):
    c = grp["y1"].corr(grp["y2"])
    corr_mat.loc[t1, t2] = c
    corr_mat.loc[t2, t1] = c
    count_mat.loc[t1, t2] = len(grp)
    count_mat.loc[t2, t1] = len(grp)
corr_mat.to_csv(os.path.join(OUT, "cross_target_corr.csv"))
count_mat.to_csv(os.path.join(OUT, "cross_target_pairs.csv"))
print("\nCross-target correlation (shared molecules):")
print(corr_mat.round(2).to_string())

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets)
ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets)
for i in range(len(targets)):
    for j in range(len(targets)):
        v = corr_mat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Cross-target correlation (shared molecules)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cross_target_corr.png")); plt.close(fig)

# PI1M quick look
pi = pd.read_csv("official_dataset/PI1M.csv", nrows=200000)
print(f"\nPI1M shape (first 200k): {pi.shape}")
print("cols:", list(pi.columns))
print(pi.head(3).to_string())
