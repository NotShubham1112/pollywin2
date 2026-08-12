import json
p = r'd:\Parth\ploywin r2\polymer_prediction_notebook.ipynb'
nb = json.load(open(p, encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if 'CB_TASK_TYPE' in src:
        c['source'] = c['source'][0].replace(
            'CB_TASK_TYPE = "GPU" if GPU_AVAILABLE else "CPU"',
            'CB_TASK_TYPE = "CPU"  # CPU-only for bit-reproducible results across runs (CatBoost GPU is nondeterministic)'
        )
        print('patched cell')
json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1)
