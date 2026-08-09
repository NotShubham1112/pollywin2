import json, sys
nb = json.load(open(r'D:\Parth\ploywin r2\PolyWin_R2_v14_p1m_pretrain.ipynb'))
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    n = len(c['source'])
    src = ''.join(c['source'])
    print(f"--- Cell {i} ({ct}, {n} lines) ---")
    if ct == 'code':
        # show fingerprint-related lines
        for line in c['source']:
            if any(k in line.lower() for k in ['morgan_r1', 'morgan_ct', 'topological', 'spectral', 'hstack', 'add_fingerprints', 'n_spectral']):
                print('  ', line.rstrip())
    print()
