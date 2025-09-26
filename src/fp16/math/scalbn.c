#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 scalb(x, y) = x * 2 ^ y;
*/
uint16_t fp16_scalbn(uint16_t x, int y) {
	return fp16_mul(x, (y + 15) << 10);
}

