#pragma once

#include "utils.h"

constexpr int SplitNum = 2;
constexpr int H = 16; // Notice: hard coded
constexpr int IntraSplitNum = 2;

#define SIMULATE_MULTIPLE 0
#define NOC_ACCL_MULTIPLE 1
constexpr int PatternLen = 2;

__device__ constexpr int8_t tile_order[IntraSplitNum][PatternLen] = {
	{0,1},
	{1,0}
	// {0,1,2,3},
};

__device__ constexpr block_iter_id srcKV[IntraSplitNum][PatternLen] = {
	{{-1, -1},{1, 0}},
	{{-1, -1},{0, 0}}, 
	// {{-1, -1},{-1, -1},{-1, -1},{-1, -1},}, 
};

__device__ constexpr block_iter_id dstKV[IntraSplitNum][PatternLen] = {
	{{1, 1},{-1, -1}},
	{{0, 1},{-1, -1}},	 
	// {{-1, -1},{-1, -1},{-1, -1},{-1, -1},}, 
};
