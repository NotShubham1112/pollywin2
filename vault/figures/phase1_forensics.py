import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

train = pd.read_csv("official_dataset/train.csv")
test = pd.read_csv("official_dataset/test.csv")
TARGETS = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]
TNAME = {"tg": "Tg (K)", "egc": "Chain Bandgap Egc", "egb": "Bulk Bandgap Egb",
         "eps": "Dielectric EPS", "nc": "Refractive Index Nc", "ei": "Ionisation Ei",
         "eea": "Electron Affinity Eea"}

# ---------- 1. Target frequency ----------
freq = train["target_type"].value_counts().rename_axis("target")
freq_df = pd.DataFrame({"count": freq, "pct": (freq / len(train) * 100).round(2)})
freq_df.to_csv(os.path.join(OUT, "target_frequency.csv"))

fig, ax = plt.subplots(figsize=(7, 4))
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(freq)))
ax.bar(freq.index, freq.values, color=colors)
for i, v in enumerate(freq.values):
    ax.text(i, v + 40, f"{v}\n({v/len(train)*100:.1f}%)", ha="center", fontsize=8)
ax.set_title("Training set size per target property")
ax.set_ylabel("samples")
ax.set_xlabel("target_type")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "target_frequency.png")); plt.close(fig)

# ---------- 2. Duplicate analysis ----------
dup_full = train.duplicated(subset=["smiles"], keep=False)
n_dup_smiles = train["smiles"].duplicated().sum()
smiles_groups = train.groupby("smiles")["target_type"].nunique()
multi_type = smiles_groups[smiles_groups > 1]

# conflicting labels: same smiles+type, different target
conflicts = train.groupby(["smiles", "target_type"])["target"].nunique()
n_conflict = (conflicts > 1).sum()

# repeated polymer families (canonical)
train["canon"] = train["smiles"].str.replace("*", "").str.replace(r"\[\*\]", "", regex=True)
dup_canon = train["canon"].duplicated().sum()
canon_groups = train.groupby("canon")["target_type"].nunique()
multi_canon = canon_groups[canon_groups > 1]

# test overlap with train
test["canon"] = test["smiles"].str.replace("*", "").str.replace(r"\[\*\]", "", regex=True)
train_canon_set = set(train["canon"].unique())
test_overlap = test["canon"].isin(train_canon_set).sum()

dup_summary = pd.DataFrame({
    "metric": ["n_unique_smiles", "n_dup_smiles", "n_rows_in_dup_groups",
               "smiles_shared_across_target_types", "conflicting_labels_same_smiles_type",
               "n_dup_canonical", "canon_shared_across_target_types",
               "test_smiles_overlapping_train"],
    "value": [train["smiles"].nunique(), n_dup_smiles, int(dup_full.sum()),
              len(multi_type), n_conflict, dup_canon, len(multi_canon), test_overlap],
})
dup_summary.to_csv(os.path.join(OUT, "duplicate_summary.csv"), index=False)
dup_summary.to_string(index=False)

# ---------- 3. Target distribution stats + transform suitability ----------
rows = []
for t in TARGETS:
    y = train.loc[train["target_type"] == t, "target"]
    rows.append({
        "target": t, "n": len(y), "mean": y.mean(), "median": y.median(),
        "std": y.std(), "skew": stats.skew(y), "kurtosis": stats.kurtosis(y),
        "min": y.min(), "max": y.max(),
        "skew_log10": stats.skew(np.log10(y - y.min() + 1e-6)),
        "shapiro_p": stats.shapiro(y.sample(min(5000, len(y))))[1],
    })
dist_df = pd.DataFrame(rows).round(4)
dist_df.to_csv(os.path.join(OUT, "target_distributions.csv"), index=False)

fig, axes = plt.subplots(4, 2, figsize=(10, 11))
for ax, t in zip(axes.ravel()[:7], TARGETS):
    y = train.loc[train["target_type"] == t, "target"]
    ax.hist(y, bins=50, color="#2a6fb0", edgecolor="white", alpha=0.9)
    ax.axvline(y.mean(), color="red", ls="--", lw=1, label=f"mean={y.mean():.2g}")
    ax.axvline(y.median(), color="orange", ls="--", lw=1, label=f"med={y.median():.2g}")
    ax.set_title(f"{t} | {TNAME[t]}")
    ax.legend(fontsize=7)
axes.ravel()[7].axis("off")
fig.suptitle("Raw target distributions (train)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "target_histograms.png")); plt.close(fig)

# log-transform suitability: compare skew before/after
fig, ax = plt.subplots(figsize=(7, 4))
sk = [stats.skew(train.loc[train["target_type"] == t, "target"]) for t in TARGETS]
skl = [stats.skew(np.log10(train.loc[train["target_type"] == t, "target"] - train.loc[train["target_type"] == t, "target"].min() + 1e-6)) for t in TARGETS]
x = np.arange(len(TARGETS))
ax.plot(x, np.abs(sk), "o-", label="|skew| raw")
ax.plot(x, np.abs(skl), "s--", label="|skew| log10-shift")
ax.set_xticks(x); ax.set_xticklabels(TARGETS)
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_title("Transform suitability: absolute skew raw vs log10")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "transform_skew.png")); plt.close(fig)

# ---------- 4. Test distribution shift (molecular weight / atom counts) ----------
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def mol_weights(smiles_list):
    ws, atoms = [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s.replace("*", "[*]").replace("*", "[*]") if False else s)
        if m is None:
            try:
                m = Chem.MolFromSmiles(s.replace("*", "[*]"))
            except Exception:
                pass
        if m is None:
            ws.append(np.nan); atoms.append(np.nan)
        else:
            ws.append(Descriptors.MolWt(m))
            atoms.append(m.GetNumHeavyAtoms())
    return np.array(ws), np.array(atoms)

for df, tag in [(train, "train"), (test, "test")]:
    w, a = mol_weights(df["smiles"].tolist())
    df["_mw"] = w; df["_ha"] = a
    df.to_csv(os.path.join("$TEMP", f"pw_{tag}.csv").replace("$TEMP", os.environ.get("TEMP", ".")), index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for tag, col in [("train", "#2a6fb0"), ("test", "#d1495b")]:
    df = train if tag == "train" else test
    axes[0].hist(df["_mw"].dropna(), bins=60, alpha=0.55, label=tag, color=col)
    axes[1].hist(df["_ha"].dropna(), bins=60, alpha=0.55, label=tag, color=col)
axes[0].set_title("Molecular weight distribution"); axes[0].legend()
axes[1].set_title("Heavy atom count"); axes[1].legend()
fig.suptitle("Train vs Test distribution shift")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "test_shift_mw.png")); plt.close(fig)

# KS test on MW by target
ks_rows = []
for t in TARGETS:
    a = train.loc[train["target_type"] == t, "_mw"].dropna()
    b = test.loc[test["target_type"] == t, "_mw"].dropna()
    if len(a) > 5 and len(b) > 5:
        ks = stats.ks_2samp(a, b)
        ks_rows.append({"target": t, "n_train": len(a), "n_test": len(b),
                        "mean_mw_train": round(a.mean(), 1), "mean_mw_test": round(b.mean(), 1),
                        "ks_stat": round(ks.statistic, 4), "ks_p": round(ks.pvalue, 6)})
ks_df = pd.DataFrame(ks_rows)
ks_df.to_csv(os.path.join(OUT, "test_shift_ks.csv"), index=False)
print("\nTest shift KS test on molecular weight:")
print(ks_df.to_string(index=False))

print("\nTarget frequency table:")
print(freq_df.to_string())
print("\nDuplicate summary table:")
print(dup_summary.to_string(index=False))
print("\nTarget distribution table:")
print(dist_df.to_string(index=False))
print("\nDONE phase1")
