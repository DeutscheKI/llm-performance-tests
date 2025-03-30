import json
import glob
from tabulate import tabulate

model_prefix = 'qwq-32b-q4_k_m '
data = []

for file in sorted(glob.glob(model_prefix+"*.json")):
    with open(file, 'r') as f:
        d = json.load(f)
        median_ttft_ms = d['median_ttft_ms']
        median_tpot_ms = d['median_tpot_ms']
        median_otps = 1000.0 / median_tpot_ms
        median_itl_ms = d['median_itl_ms']
        data.append({
            'name': file.replace(model_prefix,''),
            'median_ttft_ms': median_ttft_ms,
            'median_tpot_ms': median_tpot_ms,
            'median_itl_ms': median_itl_ms,
            'median_otps': median_otps,
        })

data.sort(key=lambda x: x['median_otps'])

print(tabulate(data, headers="keys"))
