import os, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
m = 'shubhamkambli11/polywin-r2-v17-sib-phys'
out_dir = r'D:\Parth\ploywin r2\vault\kernel-v17-sib-phys\out'
os.makedirs(out_dir, exist_ok=True)

raw = api.kernels_logs(m)
with open(os.path.join(out_dir, 'kernel_log.json'), 'w', encoding='utf-8', errors='replace') as f:
    f.write(raw)

# Parse and print meaningful traceback lines
import json
try:
    entries = json.loads(raw)
except Exception as e:
    print('parse err', e)
    entries = []

texts = []
for e in entries:
    stream = e.get('stream_name', e.get('streamName', ''))
    data = e.get('data', '')
    texts.append(data)

# Only print stderr entries and the tail of stdout
is_err = [e for e in entries if e.get('stream_name') == 'stderr' or e.get('streamName') == 'stderr']
print('=== STDERR (%d entries) ===' % len(is_err))
for e in is_err[-80:]:
    print(e.get('data', ''), end='')

print('\n=== LAST STDOUT 120 lines ===')
std = [e for e in entries if e.get('stream_name') == 'stdout' or e.get('streamName') == 'stdout']
for e in std[-120:]:
    print(e.get('data', ''), end='')