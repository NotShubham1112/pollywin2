#!/usr/bin/env python3
"""Reorganize the PolyWin R2 repository into a clean folder structure."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_git_mv(src, dst):
    """Move a tracked file using git mv, creating directories as needed."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'mv', str(src), str(dst)], cwd=ROOT, check=True)
    print(f"git mv: {src.name} -> {dst}")

def move_untracked(src, dst):
    """Move an untracked file."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"move: {src.name} -> {dst}")

def create_dirs():
    """Create the new directory structure."""
    dirs = [
        'src/core', 'src/v22_bert_arm', 'src/v20_smiles_encoder',
        'src/v13_specialist', 'src/experiments', 'src/pipeline',
        'src/notebook_builders', 'src/v2_validation',
        'notebooks/v9_gnn', 'notebooks/v10_pretrain', 'notebooks/v11_reproduce',
        'notebooks/v12_bucket_moe', 'notebooks/v13_blend', 'notebooks/v13_specialist',
        'notebooks/v14_p14_baseline', 'notebooks/v15_epsnc_focus',
        'notebooks/v16_cross_target', 'notebooks/v21_sibling_arm',
        'notebooks/v22_bert_arm', 'notebooks/exploration',
        'competition/rules', 'competition/data/raw', 'competition/data/archive',
        'docs/strategy', 'docs/design'
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"Created dir: {d}")

def move_tracked_files():
    """Move git-tracked files using git mv."""
    # Source files to src/core/
    run_git_mv(ROOT / 'mt_gnn_v2.py', ROOT / 'src/core/mt_gnn_v2.py')
    run_git_mv(ROOT / 'decoder_v16.py', ROOT / 'src/core/decoder_v16.py')
    
    # v13 specialist files
    run_git_mv(ROOT / 'v13_blend.py', ROOT / 'src/experiments/v13_blend.py')
    
    # Build scripts
    for nb in ['build_gnn_kaggle_nb.py', 'build_pipeline_nb.py', 'build_pretrain_kaggle_nb.py',
               'build_v12_kaggle_nb.py', 'build_v13_kaggle_nb.py', 'build_v14_kaggle_nb.py',
               'build_v15_kaggle_nb.py', 'build_v16_kaggle_nb.py', 'build_v21_kaggle_nb.py']:
        run_git_mv(ROOT / nb, ROOT / 'src/notebook_builders/' / nb)
    
    # Experiments
    for f in ['gnn_arm_run.py', 'gnn_moe_blend.py', 'gnn_scaffold_val.py', 'gnn_v10_blend.py',
              'r2_test_feat.pkl', 'r2_train_feat.pkl']:
        run_git_mv(ROOT / f, ROOT / 'src/experiments/' / f)
    
    # Pipeline utilities
    for f in ['make_smoke.py', 'patch_nb.py', 'restore_cell14.py', 'restore_cell2.py',
              'verify_nb.py', 'notebook_cells.txt']:
        run_git_mv(ROOT / f, ROOT / 'src/pipeline/' / f)
    
    # Old v15 notebook
    run_git_mv(ROOT / 'PolyWin_R2_v15_epsnc_focus.ipynb', 
               ROOT / 'notebooks/v15_epsnc_focus/PolyWin_R2_v15_epsnc_focus.ipynb')
    
    # v16 notebook  
    run_git_mv(ROOT / 'PolyWin_R2_v16_cross_target_decoder.ipynb',
               ROOT / 'notebooks/v16_cross_target/PolyWin_R2_v16_cross_target_decoder.ipynb')
    
    # v21 notebooks
    run_git_mv(ROOT / 'PolyWin_R2_v21_sibling_arm.ipynb',
               ROOT / 'notebooks/v21_sibling_arm/PolyWin_R2_v21_sibling_arm.ipynb')
    run_git_mv(ROOT / 'PolyWin_R2_v21_sibling_arm_smoke.ipynb',
               ROOT / 'notebooks/v21_sibling_arm/PolyWin_R2_v21_sibling_arm_smoke.ipynb')
    
    # Root pipeline notebook
    run_git_mv(ROOT / 'AISEHack_Round2_Pipeline.ipynb',
               ROOT / 'notebooks/exploration/AISEHack_Round2_Pipeline.ipynb')

def move_untracked_files():
    """Move untracked files."""
    # v20 source files
    for f in ['v20_codec.py', 'v20_encoder.py', 'v20_arm_cv.py', 'v20_blend.py',
              'v20_gate_report.py', 'run_v20_gate.py', 'build_v20_kaggle_nb.py']:
        if (ROOT / f).exists():
            move_untracked(ROOT / f, ROOT / 'src/v20_smiles_encoder/' / f)
    
    # v2 directory
    if (ROOT / 'v2').exists():
        move_untracked(ROOT / 'v2', ROOT / 'src/v2_validation')
    
    # v2 test features
    for f in ['r2_test_feat.pkl', 'r2_train_feat.pkl']:
        if (ROOT / f).exists():
            move_untracked(ROOT / f, ROOT / 'vault/' / f)
    
    # Data files
    if (ROOT / 'official_dataset').exists():
        for f in (ROOT / 'official_dataset').iterdir():
            if f.is_file():
                move_untracked(f, ROOT / 'competition/data/raw' / f.name)
        if (ROOT / 'official_dataset' / 'archive').exists():
            shutil.move(str(ROOT / 'official_dataset/archive'), 
                       str(ROOT / 'competition/data/archive'))
    
    # Rules
    if (ROOT / 'hackathon_rules').exists():
        for f in (ROOT / 'hackathon_rules').iterdir():
            if f.is_file() and f.suffix == '.md':
                move_untracked(f, ROOT / 'competition/rules' / f.name)
    
    # Strategy docs
    for f in ['polymer_modeling_strategy.md', 'polymer_research_plan.md', 
              'codex.md', 'deep-research-report (3).md', 'polymer_prediction_notebook.ipynb',
              'smoke_nb.ipynb']:
        if (ROOT / f).exists():
            move_untracked(ROOT / f, ROOT / 'docs/strategy' / f)
    
    # Docs from docs/superpowers/plans
    src_plan_dirs = ['docs/superpowers/plans/2026-08-09-v20-self-trained-smiles-encoder.md',
                     'docs/superpowers/plans/2026-08-11-v22-bert-arm.md']
    for f in src_plan_dirs:
        if (ROOT / f).exists():
            move_untracked(ROOT / f, ROOT / 'docs/design' / Path(f).name)
    
    # "New folder/" contents - the v22 code
    if (ROOT / 'New folder').exists():
        new_folder = ROOT / 'New folder'
        for f in new_folder.iterdir():
            if f.is_file():
                if f.suffix == '.py' or f.name.startswith('build_') or f.name.startswith('run_') or f.name.startswith('v22_'):
                    move_untracked(f, ROOT / 'src/v22_bert_arm' / f.name)
                elif f.suffix == '.ipynb':
                    dest_dir = ROOT / 'notebooks/v22_bert_arm'
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    move_untracked(f, dest_dir / f.name)
                elif f.suffix == '.md' or (f.suffix == '.' if hasattr(f, 'suffix') else False):
                    # Markdown/design docs
                    dest_dir = ROOT / 'docs/design'
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    move_untracked(f, dest_dir / f.name)
                elif f.name.startswith('test_'):
                    move_untracked(f, ROOT / 'tests' / f.name)
        
        # Move "New folder/tests/" contents
        tests_dir = new_folder / 'tests'
        if tests_dir.exists():
            for f in tests_dir.iterdir():
                if f.is_file() and f.suffix == '.py':
                    move_untracked(f, ROOT / 'tests' / f.name)
        
        # Move "New folder/v2/" if it exists there
        v2_dir = new_folder / 'v2'
        if v2_dir.exists():
            move_untracked(v2_dir, ROOT / 'src/v2_validation')
        
        # Move notebooks subfolder
        notebooks_sub = new_folder / 'notebooks'
        if notebooks_sub.exists():
            for nb_dir in notebooks_sub.iterdir():
                if nb_dir.is_dir():
                    dest_dir = ROOT / 'notebooks' / nb_dir.name
                    for nb in nb_dir.iterdir():
                        if nb.is_file():
                            move_untracked(nb, dest_dir / nb.name)
        
        # Remove empty "New folder"
        try:
            new_folder.rmdir()
        except:
            pass

def update_path_references():
    """Update all path references in files."""
    # Path mappings for replacement
    path_updates = [
        ('"src/core/mt_gnn_v2.py"', '"src/core/mt_gnn_v2.py"'),
        ('"src/core/decoder_v16.py"', '"src/core/decoder_v16.py"'),
        ('"competition/data/raw"', '"competition/data/raw"'),
        ('"src/notebook_builders/build_v14_kaggle_nb.py"', '"src/notebook_builders/build_v14_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v20_kaggle_nb.py"', '"src/notebook_builders/build_v20_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v21_kaggle_nb.py"', '"src/notebook_builders/build_v21_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v16_kaggle_nb.py"', '"src/notebook_builders/build_v16_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v13_kaggle_nb.py"', '"src/notebook_builders/build_v13_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v12_kaggle_nb.py"', '"src/notebook_builders/build_v12_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v15_kaggle_nb.py"', '"src/notebook_builders/build_v15_kaggle_nb.py"'),
        ('"src/v22_bert_arm/build_v22_kaggle_nb.py"', '"src/v22_bert_arm/build_v22_kaggle_nb.py"'),
        ('"notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb"', '"notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb"'),
        ('"notebooks/v11_reproduce/PolyWin_R2_v11_reproduce.ipynb"', '"notebooks/v11_reproduce/PolyWin_R2_v11_reproduce.ipynb"'),
        ('"notebooks/v12_bucket_moe/PolyWin_R2_v12_bucket_moe.ipynb"', '"notebooks/v12_bucket_moe/PolyWin_R2_v12_bucket_moe.ipynb"'),
        ('"notebooks/v13_blend/PolyWin_R2_v13_gbm_gnn_blend.ipynb"', '"notebooks/v13_blend/PolyWin_R2_v13_gbm_gnn_blend.ipynb"'),
        ('"notebooks/v15_epsnc_focus/PolyWin_R2_v15_epsnc_focus.ipynb"', '"notebooks/v15_epsnc_focus/PolyWin_R2_v15_epsnc_focus.ipynb"'),
        ('"notebooks/v16_cross_target/PolyWin_R2_v16_cross_target_decoder.ipynb"', '"notebooks/v16_cross_target/PolyWin_R2_v16_cross_target_decoder.ipynb"'),
        ('"notebooks/v21_sibling_arm/PolyWin_R2_v21_sibling_arm.ipynb"', '"notebooks/v21_sibling_arm/PolyWin_R2_v21_sibling_arm.ipynb"'),
        ('"notebooks/v22_bert_arm/PolyWin_R2_v22_bert_arm.ipynb"', '"notebooks/v22_bert_arm/PolyWin_R2_v22_bert_arm.ipynb"'),
        ('"notebooks/v9_gnn/PolyWin_R2_v9_GNN_kaggle.ipynb"', '"notebooks/v9_gnn/PolyWin_R2_v9_GNN_kaggle.ipynb"'),
        ('"src/notebook_builders/build_v14_kaggle_nb.py"', '"src/notebook_builders/build_v14_kaggle_nb.py"'),
        ('"src/notebook_builders/build_v11_reproduce_nb.py"', '"src/notebook_builders/build_v11_reproduce_nb.py"'),
    ]
    
    # Find all Python and Markdown files
    extensions = ['*.py', '*.md', '*.ipynb']
    files_to_update = []
    for ext in extensions:
        files_to_update.extend(ROOT.rglob(ext))
    
    # Filter out vault/ and venv
    files_to_update = [f for f in files_to_update 
                       if 'vault' not in str(f) and '.venv' not in str(f) 
                       and '.kilo' not in str(f) and '.superpowers' not in str(f)]
    
    for f in files_to_update:
        try:
            content = f.read_text(encoding='utf-8', errors='ignore')
            original = content
            for old, new in path_updates:
                content = content.replace(old, new)
            if content != original:
                f.write_text(content, encoding='utf-8')
                print(f"Updated paths in: {f.name}")
        except Exception as e:
            print(f"Could not update {f}: {e}")

def generate_readme():
    """Generate a comprehensive README.md."""
    readme = '''# PolyWin R2 — AISEHack 2.0 Polymer Property Prediction Round 2

**Kaggle Competition**: [PPP Round 2 Rules](https://www.kaggle.com/competitions/ppp-round-2/rules)  
**Targets**: EEA, EGB, EGC, EI, EPS, NC, Tg (7 polymer properties)  
**Metric**: Mean R² across all targets  
**Best LB Score**: 0.883 (P14 baseline submitted 2026-08-10)

## Repository Structure

```
├── README.md                          # This file
├── .gitignore                         # Standard Python + Kaggle ignores
│
├── competition/                       # Competition data and rules
│   ├── rules/                         # AISEHack rules documentation
│   └── data/
│       ├── raw/                       # train.csv, test.csv, PI1M.csv, sample_submission.csv
│       └── archive/                   # Historical data
│
├── src/                               # All source code
│   ├── core/                          # Core shared source files
│   │   ├── mt_gnn_v2.py              # P14 main source (MT-GNN with GBM stack)
│   │   └── decoder_v16.py            # Cross-target decoder (v16)
│   │
│   ├── v22_bert_arm/                  # Latest experiment: BERT SMILES encoder
│   │   ├── v22_tokenizer.py
│   │   ├── v22_encoder.py
│   │   ├── v22_arm_cv.py
│   │   ├── v22_blend.py
│   │   ├── v22_gate_report.py
│   │   ├── build_v22_kaggle_nb.py
│   │   ├── run_v22_gate.py
│   │   └── tests/                     # v22 unit tests
│   │
│   ├── v20_smiles_encoder/            # Self-trained SMILES encoder (v20)
│   │   ├── v20_codec.py
│   │   ├── v20_encoder.py
│   │   ├── v20_arm_cv.py
│   │   ├── v20_blend.py
│   │   ├── v20_gate_report.py
│   │   ├── run_v20_gate.py
│   │   └── build_v20_kaggle_nb.py
│   │
│   ├── notebook_builders/             # Build Kaggle notebooks from source
│   │   ├── build_v14_kaggle_nb.py     # P14 final submission
│   │   ├── build_v16_kaggle_nb.py     # Cross-target decoder
│   │   ├── build_v20_kaggle_nb.py     # Self-trained encoder
│   │   ├── build_v21_kaggle_nb.py     # Sibling arm
│   │   ├── build_v12_kaggle_nb.py     # Bucket MoE
│   │   ├── build_v13_kaggle_nb.py     # Specialist blend
│   │   ├── build_v15_kaggle_nb.py     # EPS/NC focus
│   │   ├── build_gnn_kaggle_nb.py
│   │   ├── build_pipeline_nb.py
│   │   └── build_pretrain_kaggle_nb.py
│   │
│   ├── v13_specialist/
│   ├── v2_validation/                 # Original v2/ directory
│   ├── experiments/                   # Experiment scripts
│   └── pipeline/                      # Pipeline utilities
│
├── notebooks/                         # Experiment notebooks by version
│   ├── v14_p14_baseline/             # FINAL: P14 submission (0.883 LB)
│   ├── v22_bert_arm/                 # Latest: BERT encoder on P14 blend
│   ├── v21_sibling_arm/              # Leak-safe sibling features
│   ├── v20_smiles_encoder/           # Self-trained SMILES representation
│   ├── v16_cross_target/             # Cross-target decoder
│   ├── v15_epsnc_focus/              # EPS/NC targeted improvements
│   ├── v14_p1m_pretrain/             # P14 with PI1M pretraining
│   ├── v13_blend/                    # Specialist modeling
│   ├── v12_bucket_moe/               # Mixture of Experts
│   ├── v11_reproduce/                # Reproduction experiments
│   ├── v10_pretrain/                 # Pretraining experiments
│   ├── v9_gnn/                       # Early GNN experiments
│   └── exploration/                  # Strategy notebooks, pipeline
│
├── tests/                             # All unit tests
│   ├── test_{v14,v16,v20,v21,v12,v13,v15}.py
│   ├── test_v22_*.py                 # v22 BERT arm tests
│   └── ...
│
├── docs/
│   ├── strategy/                     # Strategy & research plans
│   ├── design/                       # Design specs & implementation plans
│   └── superpowers/                  # Kilo agent documentation
│
├── vault/                             # Archive of experiments, outputs, backups
│   └── (thousands of files)
│
└── submission_v17_final.csv           # Final submission (v17)
`` '''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    print("Generated README.md")

if __name__ == '__main__':
    print("Creating directory structure...")
    create_dirs()
    
    print("\nMoving tracked files with git mv...")
    move_tracked_files()
    
    print("\nMoving untracked files...")
    move_untracked_files()

    print("\nUpdating path references...")
    update_path_references()

    print("\nGenerating README.md...")
    generate_readme()

    print("\nDone! Run 'git status' to review changes.")