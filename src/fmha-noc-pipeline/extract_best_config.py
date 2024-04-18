import re

b, seqlen, hdim, heads = 1, 4096, 512, 16
flops = 4 * b * heads * seqlen * seqlen * hdim

def extract_data(log_filename):
    results = []  # 存储所有组的数据
    # 当前大组数据
    last_simulate_multiple = None
    current_simulate_multiple = None
    # 当前小组数据
    current_noc_accl_multiple = None
    current_min_ms_value = float('inf')
    current_min_pattern_len = None
    start = True

    with open(log_filename, 'r') as file:
        for line in file:                   
            if 'noc_accl_multiple=' in line:
                if not start:
                    results.append({
                        'simulate_multiple': last_simulate_multiple,
                        'noc_accl_multiple': current_noc_accl_multiple,
                        'min_ms_value': current_min_ms_value,
                        'pattern_len': current_min_pattern_len
                    })
                last_simulate_multiple = current_simulate_multiple
                current_noc_accl_multiple = int(re.search(r'noc_accl_multiple=(\d+)', line).group(1))
                current_min_ms_value = float('inf')
                current_min_pattern_len = None
                start = False
                continue
            
            if 'simulate_multiple=' in line:
                current_simulate_multiple = int(re.search(r'simulate_multiple=(\d+)', line).group(1))
                continue

            pattern_match = re.search(r'pattern_len=(\d+)', line)
            if pattern_match:
                current_pattern_len = int(pattern_match.group(1))
            
            ms_value_match = re.search(r'ms_value: ([\d\.]+)', line)
            if ms_value_match:
                ms_value = float(ms_value_match.group(1))
                if ms_value < current_min_ms_value:
                    current_min_ms_value = ms_value
                    current_min_pattern_len = current_pattern_len

        results.append({
            'simulate_multiple': last_simulate_multiple,
            'noc_accl_multiple': current_noc_accl_multiple,
            'min_ms_value': current_min_ms_value,
            'pattern_len': current_min_pattern_len
        })

    return results

log_filename = '../../logs/script_log.txt'
results = extract_data(log_filename)
for data in results:
    print(f"simulate_multiple: {data['simulate_multiple']}, noc_accl_multiple: {data['noc_accl_multiple']}, "
          f"gflops: {int(flops / (data['min_ms_value'] * 1e6))}, noc_ratio: {1 / data['pattern_len']}")
