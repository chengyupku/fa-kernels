import sys

SplitNum = 2
H = 16
IntraSplitNum = 2

def generate_configs(pattern_len, simulate_multiple, noc_accl_multiple):
#     header = f"""#pragma once

# #include "utils.h"

# constexpr int SplitNum = 2;
# constexpr int H = 16; // Notice: hard coded
# constexpr int IntraSplitNum = 2;

# #define SIMULATE_MULTIPLE {simulate_multiple}
# #define NOC_ACCL_MULTIPLE {noc_accl_multiple}
# constexpr int PatternLen = {pattern_len};
#     """
#     tile_order_decl = "\n__device__ constexpr int8_t tile_order[IntraSplitNum][PatternLen] = {"
#     for tidx in range(IntraSplitNum):
#         tile_order_decl += "\n\t{"
#         tile_order_decl += f"{tidx},{(tidx + 1) % IntraSplitNum},"
#         for i in range(2, pattern_len):
#             tile_order_decl += f"{i},"
#         tile_order_decl += "},"
#     tile_order_decl += "\n};\n"

#     src_decl = "\n__device__ constexpr block_iter_id srcKV[IntraSplitNum][PatternLen] = {"
#     for tidx in range(IntraSplitNum):
#         src_decl += "\n\t{"
#         src_decl += "{-1, -1},"
#         src_decl += f"{{{(tidx + 1) % IntraSplitNum}, 0}},"
#         for i in range(pattern_len - 2):
#             src_decl += "{-1, -1},"
#         src_decl += "}, "
#     src_decl += "\n};\n"

#     dst_decl = "\n__device__ constexpr block_iter_id dstKV[IntraSplitNum][PatternLen] = {"
#     for tidx in range(IntraSplitNum):
#         dst_decl += "\n\t{"
#         dst_decl += f"{{{(tidx + 1) % IntraSplitNum}, 1}},"
#         for i in range(pattern_len - 1):
#             dst_decl += "{-1, -1},"
#         dst_decl += "}, "
#     dst_decl += "\n};\n"

#     code = ""
#     code += header
#     code += tile_order_decl
#     code += src_decl
#     code += dst_decl

    code = f"""#pragma once

#include "utils.h"

constexpr int SplitNum = 2;
constexpr int H = 16; // Notice: hard coded
constexpr int IntraSplitNum = 2;

#define SIMULATE_MULTIPLE {simulate_multiple}
#define NOC_ACCL_MULTIPLE 1
constexpr int PatternLen = 2;
"""
    code += """
__device__ constexpr int8_t tile_order[IntraSplitNum][PatternLen] = {
	{0,1},
	{0,1},
};

__device__ constexpr block_iter_id srcKV[IntraSplitNum][PatternLen] = {
	{{-1, -1},{-1, -1},}, 
	{{-1, -1},{-1, -1},}, 
};

__device__ constexpr block_iter_id dstKV[IntraSplitNum][PatternLen] = {
	{{-1, -1},{-1, -1},}, 
	{{-1, -1},{-1, -1},}, 
};
"""
    return code

if __name__ == "__main__":
    args = sys.argv[1:]
    pattern_len = int(args[0])
    simulate_multiple = int(args[1])
    noc_accl_multiple = int(args[2])
    with open("noc_config.h", "w") as f:
        f.write(generate_configs(pattern_len, simulate_multiple, noc_accl_multiple))