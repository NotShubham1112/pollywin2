import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from collections import Counter
import os

OUT = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv("official_dataset/train.csv")
TARGETS = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]

def parse_mol(smiles):
    m = Chem.MolFromSmiles(smiles.replace("*", "[*]"))
    if m is None:
        m = Chem.MolFromSmiles(smiles.replace("*", "C"))
    return m

def polymer_features(smiles):
    m = parse_mol(smiles)
    if m is None:
        return {k: np.nan for k in ["aromatic_ratio", "hetero_ratio", "ring_count", "ring_ratio",
                                    "nF", "nSi", "nS", "nN", "nO", "nCl", "aromatic_atoms", "heavy",
                                    "conj_bonds", "rot_bonds", "frac_sp3", "donors", "acceptors",
                                    "polarizability", "logP", "n_conj_rings", "linear_chain", "MW"]}
    arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    heavy = m.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(m)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(m)
    feats = {
        "aromatic_ratio": arom / max(heavy, 1),
        "hetero_ratio": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6) and a.GetAtomicNum() != 0) / max(heavy, 1),
        "ring_count": rings,
        "ring_ratio": rings / max(heavy, 1) * 2,
        "nF": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 9),
        "nSi": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 14),
        "nS": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 16),
        "nN": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7),
        "nO": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 8),
        "nCl": sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 17),
        "aromatic_atoms": arom,
        "heavy": heavy,
        "conj_bonds": rdMolDescriptors.CalcNumAromaticRings(m) + sum(1 for b in m.GetBonds() if b.GetIsAromatic()),
        "rot_bonds": rot_bonds,
        "frac_sp3": Descriptors.FractionCSP3(m),
        "donors": rdMolDescriptors.CalcNumHBD(m),
        "acceptors": rdMolDescriptors.CalcNumHBA(m),
        "polarizability": Descriptors.MolMR(m),
        "logP": Descriptors.MolLogP(m),
        "n_conj_rings": rdMolDescriptors.CalcNumAromaticRings(m),
        "linear_chain": float(rot_bonds >= 4 and rings <= 1),
        "MW": Descriptors.MolWt(m),
    }
    return feats

feat_cols = ["aromatic_ratio", "hetero_ratio", "ring_count", "ring_ratio", "nF", "nSi", "nS",
             "nN", "nO", "nCl", "aromatic_atoms", "heavy", "conj_bonds", "rot_bonds", "frac_sp3",
             "donors", "acceptors", "polarizability", "logP", "n_conj_rings", "linear_chain", "MW"]

F = train["smiles"].apply(polymer_features)
F = pd.DataFrame(F.tolist(), index=train.index)
train = pd.concat([train, F], axis=1)

# Family taxonomy flags
train["fam_fluorinated"] = train["nF"] > 0
train["fam_silicon"] = train["nSi"] > 0
train["fam_sulfur"] = train["nS"] > 0
train["fam_conjugated"] = train["n_conj_rings"] >= 1
train["fam_ring_rich"] = (train["ring_count"] >= 3) & (train["heavy"] >= 15)
train["fam_linear"] = train["linear_chain"] == 1
train["fam_aromatic"] = train["aromatic_ratio"] > 0.3
train["fam_hetero"] = train["hetero_ratio"] > 0.15
train["fam_rigid"] = (train["rot_bonds"] <= 2) & (train["ring_count"] >= 2)

fam_cols = ["fam_fluorinated", "fam_silicon", "fam_sulfur", "fam_conjugated", "fam_ring_rich",
            "fam_linear", "fam_aromatic", "fam_hetero", "fam_rigid"]
fam_df = pd.DataFrame({"family": fam_cols,
                       "n": train[fam_cols].sum().values,
                       "pct": (train[fam_cols].mean() * 100).round(1).values})
fam_df["family"] = fam_df["family"].str.replace("fam_", "")
fam_df.to_csv(os.path.join(OUT, "polymer_families.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(fam_df["family"], fam_df["pct"], color=plt.cm.plasma(np.linspace(0.1, 0.9, len(fam_df))))
for b, v in zip(bars, fam_df["pct"]):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% of dataset")
ax.set_title("Polymer family taxonomy (train set)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "polymer_families.png")); plt.close(fig)

# Chemistry stats per target
chem_by_target = train.groupby("target_type")[feat_cols].mean().round(3)
chem_by_target.to_csv(os.path.join(OUT, "chemistry_by_target.csv"))

# F / Si / S / hetero / aromatic distribution across targets
summary_rows = []
for t in TARGETS:
    sub = train[train["target_type"] == t]
    summary_rows.append({
        "target": t, "n": len(sub),
        "pct_aromatic": round(sub["fam_aromatic"].mean() * 100, 1),
        "pct_fluorinated": round(sub["fam_fluorinated"].mean() * 100, 1),
        "pct_silicon": round(sub["fam_silicon"].mean() * 100, 1),
        "pct_sulfur": round(sub["fam_sulfur"].mean() * 100, 1),
        "pct_conjugated": round(sub["fam_conjugated"].mean() * 100, 1),
        "pct_linear": round(sub["fam_linear"].mean() * 100, 1),
        "pct_rigid": round(sub["fam_rigid"].mean() * 100, 1),
        "avg_aromatic_ratio": round(sub["aromatic_ratio"].mean(), 3),
        "avg_ring_count": round(sub["ring_count"].mean(), 2),
        "avg_nF": round(sub["nF"].mean(), 2),
        "avg_nSi": round(sub["nSi"].mean(), 2),
    })
summ = pd.DataFrame(summary_rows)
summ.to_csv(os.path.join(OUT, "chemistry_summary.csv"), index=False)

# Correlation of chemistry features with target (Spearman) - only within-target
corr_rows = []
for t in TARGETS:
    sub = train[train["target_type"] == t]
    for c in feat_cols:
        if sub[c].nunique() < 3:
            continue
        rho = sub[c].corr(sub["target"], method="spearman")
        corr_rows.append({"target": t, "feature": c, "spearman": round(rho, 3)})
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(OUT, "chemistry_target_corr.csv"), index=False)

# heatmap
piv = corr_df.pivot(index="feature", columns="target", values="spearman")
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
ax.set_title("Spearman correlation: chemistry feature → target")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "chemistry_heatmap.png")); plt.close(fig)

# top features per target (abs spearman)
print("Top-3 chemistry drivers per target:")
for t in TARGETS:
    top = corr_df[corr_df["target"] == t].reindex(corr_df[corr_df["target"] == t]["spearman"].abs().sort_values(ascending=False).index).head(3)
    print(t, "→", ", ".join(f"{r['feature']}={r['spearman']:.2f}" for _, r in top.iterrows()))

print("\nPolymer family taxonomy:")
print(fam_df.to_string(index=False))
print("\nChemistry summary by target:")
print(summ.to_string(index=False))
print("\nDONE phase2")
