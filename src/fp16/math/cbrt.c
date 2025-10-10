#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 cuberoot
 
 cbrt(x) = x ^ 1/3
*/
fp5x10 fp16_cbrt(fp5x10 x) {
	if((x & 0x7FFF) >= 0x7C00) //inf, nan
	 return x;
	return fp16_exp2(fp16_mul(0x3555, fp16_log2(x)));
}
