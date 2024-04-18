import subprocess
import sys
import re
import json
import time

def extract_time(filename):
    with open(filename, 'r') as file:
        cute_fmha_lines = []
        gemm_check_passed = False
        for line in file:
            if "CUTE_FMHA:" in line:
                cute_fmha_lines.append(line)
            if "gemm-check-2:" in line:
                if "Passed" in line:
                    gemm_check_passed = True
        if not gemm_check_passed:
            print("GEMM check failed.")

        if len(cute_fmha_lines) >= 2:
            ms_value = 1e8
            second_line = cute_fmha_lines[1]
            match = re.search(r'\(([\d\.]+)\)', second_line.split()[-1])
            if match:
                ms_value = float(match.group(1))
            else:
                raise ValueError("No number found")
            return ms_value
        else:
            raise ValueError("Not enough CUTE_FMHA lines found")


# log_path = "../../logs/temp_log.txt"
# best_pattern_len_dict = {}
# for simulate_multiple in range(9):
#     print("====================================================")
#     print(f"simulate_multiple={simulate_multiple}")
#     print("====================================================")
#     for noc_accl_multiple in range(1, 9):
#         print("----------------------------------------------------")
#         print(f"noc_accl_multiple={noc_accl_multiple}")
#         print("----------------------------------------------------")
#         best_pattern_len = -1
#         best_time = 1e8
#         for pattern_len in range(2, 11):
#             print(f"pattern_len={pattern_len}")
#             try:
#                 subprocess.run(f"python generate_noc_configs.py {pattern_len} {simulate_multiple} {noc_accl_multiple}", shell=True)
#             except:
#                 print("error: generate_noc_configs.py")
#                 continue

#             start_time = time.time()
#             try:
#                 subprocess.run(f"CUDA_VISIBLE_DEVICES=6 python -u compile_run_all_configs.py verify EXECMODE=1 STAGECOUNT=2 None None None None > {log_path} 2>&1", shell=True, timeout=120)
#             except subprocess.TimeoutExpired:
#                 print(f"Timeout: The process took longer than 120 seconds and was killed.")
#                 continue
#             except:
#                 print("error: compile_run_all_configs.py")
#                 continue
#             end_time = time.time()
#             duration = end_time - start_time
#             print(f"profile took {duration:.2f} seconds.")

#             try:
#                 ms_value = extract_time(log_path)
#                 print("ms_value:", ms_value)
#                 if ms_value < best_time:
#                     best_time = ms_value
#                     best_pattern_len = pattern_len
#             except Exception as e:
#                 print(f"error: {e}")
#         best_pattern_len_dict[f"simulate_multiple={simulate_multiple}, noc_accl_multiple={noc_accl_multiple}"] = (best_pattern_len, best_time)

# with open('results.json', 'w') as file:
#     json.dump(best_pattern_len_dict, file, indent=4)

log_path = "../../logs/temp_log.txt"
best_pattern_len_dict = {}
noc_accl_multiple = 1
pattern_len = 2

for simulate_multiple in range(9):
    print("====================================================")
    print(f"simulate_multiple={simulate_multiple}")
    print("====================================================")
    best_pattern_len = -1
    best_time = 1e8

    try:
        subprocess.run(f"python generate_noc_configs.py {pattern_len} {simulate_multiple} {noc_accl_multiple}", shell=True)
    except:
        print("error: generate_noc_configs.py")
        continue

    start_time = time.time()
    try:
        subprocess.run(f"CUDA_VISIBLE_DEVICES=6 python -u compile_run_all_configs.py verify EXECMODE=1 STAGECOUNT=2 None None None None > {log_path} 2>&1", shell=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"Timeout: The process took longer than 120 seconds and was killed.")
        continue
    except:
        print("error: compile_run_all_configs.py")
        continue
    end_time = time.time()
    duration = end_time - start_time
    print(f"profile took {duration:.2f} seconds.")

    try:
        ms_value = extract_time(log_path)
        print("ms_value:", ms_value)
        if ms_value < best_time:
            best_time = ms_value
            best_pattern_len = pattern_len
    except Exception as e:
        print(f"error: {e}")
    best_pattern_len_dict[f"simulate_multiple={simulate_multiple}, noc_accl_multiple={noc_accl_multiple}"] = (best_pattern_len, best_time)