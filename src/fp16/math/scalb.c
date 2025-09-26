#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 scalb(x, y) = x * 2 ^ y;
*/
uint16_t fp16_scalb(uint16_t x, uint16_t y) {
	return fp16_mul(x, fp16_exp2(y));
}

