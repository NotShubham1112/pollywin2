import io, os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Replicate kernels_output but with utf-8 writing of the log
out_dir = r'D:\Parth\ploywin r2\vault\kernel-v17-sib-phys\out'
os.makedirs(out_dir, exist_ok=True)

kernel = 'shubhamkambli11/polywin-r2-v17-sib-phys'
try:
    kernel_obj = api.kernel_metadata(kernel)
    log = api.kernel_output(kernel, path=out_dir)
    # kernel_output returns a tuple (kernel_log, kernel_stream)
    if isinstance(log, tuple):
        text, files = log
    else:
        text, files = log, {}
    log_name = kernel_obj.ref.split('/')[-1] + '.log'
    log_path = os.path.join(out_dir, log_name)
    with open(log_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(text if isinstance(text, str) else str(text))
    print('wrote log to', log_path)
    print('--- LAST 100 LINES ---')
    lines = text.splitlines() if isinstance(text, str) else str(text).splitlines()
    print('\n'.join(lines[-100:]))
except Exception as e:
    print('ERR:', repr(e))