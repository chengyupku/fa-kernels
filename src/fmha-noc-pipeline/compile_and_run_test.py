import subprocess
import sys

c_cmd = f"/usr/local/cuda-12.2/bin/nvcc -ptx --use_fast_math -forward-unknown-to-host-compiler -I../../lib/ -I ../../include -I/home/msra/mnt/fa-kernels/cutlass-3.4.0/include -I/home/msra/mnt/fa-kernels/cutlass-3.4.0/examples/common -I/usr/local/cuda-12.2/include -I/include -I/examples -I/home/msra/mnt/fa-kernels/cutlass-3.4.0/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a] -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -std=c++17 -MD -MT -MF -x cu copy_test.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o copy_test.ptx"

print("=" * 100)
print(c_cmd)
print("=" * 100)

result = subprocess.run(c_cmd, shell=True)
if result.returncode != 0:
    print("Compile Error.")

print("Compile End.")


subprocess.run(f"./copy_test", shell=True)