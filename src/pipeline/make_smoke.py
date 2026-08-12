import json
src = r'd:\Parth\ploywin r2\polymer_prediction_notebook.ipynb'
dst = r'C:\Users\shubh\AppData\Local\Temp\opencode\smoke_r2\smoke_nb.ipynb'
nb = json.load(open(src, encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c['source'])
    if 'N_FOLDS = 5' in s:
        s = s.replace('N_FOLDS = 5', 'N_FOLDS = 2')
        s = s.replace('MAX_ESTIMATORS = 3000', 'MAX_ESTIMATORS = 60')
        s = s.replace('EARLY_STOPPING_ROUNDS = 150', 'EARLY_STOPPING_ROUNDS = 20')
        c['source'] = [s]
        print('budget-reduced cell patched')
json.dump(nb, open(dst, 'w', encoding='utf-8'), indent=1)
print('wrote smoke notebook')
