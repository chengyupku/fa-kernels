import subprocess
import sys

# Parameters
q_blk_sizes = [64]
k_blk_sizes = [64]
head_sizes = [256]
head_num = 16
batch_size = 1
seq_len = 4096
# Only run fp16 inputs
precs = [1] 
verify = "verify"
verify_all = "verify-all"
verify_flags = ""
ref_check = False
input1 = sys.argv[1]
print(input1)

if input1 == verify:
    ref_check = True

if input1 == verify_all:
    verify_flags = "-DCOPYOUTMM0 -DCOPYOUTMI"
    ref_check = True

if sys.argv[2] == "CTA256":
    q_blk_sizes = [128]

for q_blk in q_blk_sizes:
    for k_blk in k_blk_sizes:
        c_cmd = f"/usr/local/cuda-12.2/bin/nvcc -D{sys.argv[2]} -D{sys.argv[3]} -D{sys.argv[4]} -D{sys.argv[5]} -D{sys.argv[6]} {verify_flags} --use_fast_math -forward-unknown-to-host-compiler -DQBLKSIZE={q_blk} -DKBLKSIZE={k_blk} -I../../lib/ -I ../../include -I/home/msra/mnt/fa-kernels/cutlass-3.4.0/include -I/home/msra/mnt/fa-kernels/cutlass-3.4.0/examples/common -I/usr/local/cuda-12.2/include -I/include -I/examples -I/home/msra/mnt/fa-kernels/cutlass-3.4.0/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a] -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -std=c++17 -MD -MT -MF -x cu fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_pipeline"

        print("=" * 100)
        print(c_cmd)
        print("=" * 100)

        result = subprocess.run(c_cmd, shell=True)
        if result.returncode != 0:
            print("Compile Error.")
            continue
        
        print("Compile End.")

        for prec in precs:
            for head in head_sizes:
                # VERIFICATION RUN
                if input1 == verify or input1 == verify_all:
                    subprocess.run(f"./fmha_forward_pipeline --batch-size={batch_size} --seq-length={seq_len} --dim-size={head_num * head} --iterations=1 --head-size={head} --reference-check=true --prec-type={prec}", shell=True)

                # # FLOP RUN
                # if input1 != verify_all:
                #     print("*" * 100)
                #     print("FLOP RUN BEGIN")
                #     print(f"PREC={prec}, QBLK={q_blk}, KBLK={k_blk}, HEAD={head}, CTA={sys.argv[2]}, {sys.argv[3]}, {sys.argv[4]}, {sys.argv[5]}, {sys.argv[6]}")
                #     subprocess.run(f"./fmha_forward_pipeline --batch-size={batch_size} --seq-length={seq_len} --dim-size={head_num * head} --iterations=1000 --head-size={head} --reference-check=false --prec-type={prec}", shell=True)
                #     print("FLOP RUN END")
                #     print("*" * 100)
