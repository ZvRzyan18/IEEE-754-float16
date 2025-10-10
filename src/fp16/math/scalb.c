#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 scalb(x, y) = x * 2 ^ y;
*/
fp5x10 fp16_scalb(fp5x10 x, fp5x10 y) {
	return fp16_mul(x, fp16_exp2(y));
}

