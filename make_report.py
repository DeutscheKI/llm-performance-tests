import json
import glob
from tabulate import tabulate
import re

model_prefix = 'QwQ-32B-AWQ '
rr_pattern = re.compile(r'rr(\d+)_(\d+)')
data = []

for file in sorted(glob.glob(model_prefix+"*.json")):
    with open(file, 'r') as f:
        d = json.load(f)
        filename = file.replace(model_prefix,'').replace('.json','')
        rr = rr_pattern.search(filename)
        request_rate = 1.0
        if rr is not None:
            request_rate = float(rr[1]+'.'+rr[2])
            filename = filename.replace(rr[0],'')
        median_ttft_ms = d['median_ttft_ms']
        median_tpot_ms = d['median_tpot_ms']
        median_otps = 1000.0 / median_tpot_ms
        median_itl_ms = d['median_itl_ms']
        data.append({
            'name': filename,
            'request_rate': request_rate,
            'median_ttft_ms': median_ttft_ms,
            'median_tpot_ms': median_tpot_ms,
            'median_itl_ms': median_itl_ms,
            'median_otps': median_otps,
        })

data.sort(key=lambda x: x['median_otps'])

print(tabulate(data, headers="keys", tablefmt="github"))
