#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

#include <fenv.h>
/*
 scalb(x, y) = x * 2 ^ y;
*/
uint16_t fp16_scalbn(uint16_t x, int y) {
	if(y > 15)
	 feraiseexcept(FE_OVERFLOW);
	if(y < -14)
	 feraiseexcept(FE_UNDERFLOW);
	return fp16_mul(x, (y + 15) << 10);
}

