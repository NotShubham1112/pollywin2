import json
nb = json.load(open(r'D:\Parth\ploywin r2\PolyWin_R2_v14_p1m_pretrain.ipynb', encoding='utf-8'))
names = []
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    names.append((i, src))

for var in ['idx_of_target', 'oof_gbm_global', 'oof_mt_global', 'final_te',
            'Y', 'G', 'Xte', 'tef', 'GLOBAL_FOLDS', 'OUT', 'stack_test', 'train',
            'stack_oof', 'mt_oof', 'mt_test']:
    found = []
    for i, s in names:
        for ln in s.splitlines():
            if ln.strip().startswith(var + ' =') or ln.strip().startswith(var + '='):
                found.append(i)
                break
    print(f'{var}: cells {found if found else "???"}')