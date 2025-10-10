#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

#include <fenv.h>
/*
 scalb(x, y) = x * 2 ^ y;
*/
fp5x10 fp16_scalbn(fp5x10 x, int y) {
	if(y > 15)
	 feraiseexcept(FE_OVERFLOW);
	if(y < -14)
	 feraiseexcept(FE_UNDERFLOW);
	return fp16_mul(x, (y + 15) << 10);
}

