#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 get the exponent of x and return it as a signed integer
*/
int fp16_ilogb(uint16_t x) {
	return ((int)x >> 10) - 15;
}

