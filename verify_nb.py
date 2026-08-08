import json, io, sys, re
nb = json.load(open(r'd:\Parth\ploywin r2\polymer_prediction_notebook.ipynb', encoding='utf-8'))
orig = open(r'D:\Parth\ploywin r2\notebook_cells.txt', encoding='utf-8').read()
blocks = re.split(r'=== code cell (\d+) ===\n', orig)
orig_map = {int(blocks[i]): blocks[i+1] for i in range(1, len(blocks)-1, 2)}
out = io.StringIO()
old_gpu = 'CB_TASK_TYPE = "GPU" if GPU_AVAILABLE else "CPU"'
new_gpu = 'CB_TASK_TYPE = "CPU"  # CPU-only for bit-reproducible results across runs (CatBoost GPU is nondeterministic)'
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if i in orig_map:
            o2 = orig_map[i].replace(old_gpu, new_gpu).rstrip('\n')
            if src.rstrip('\n') != o2:
                out.write('DIFF in cell %d (orig %d, now %d)\n' % (i, len(o2), len(src)))
            else:
                out.write('cell %d: MATCH (%d chars)\n' % (i, len(src)))
sys.stdout.buffer.write(out.getvalue().encode('utf-8'))
