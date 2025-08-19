#include "fp16/math.h"
#include "fp16/float16.h"

// 2 degree polynomial 
static const uint16_t c[3] = {
	0x8A4B,
	0x2043,
	0xB155
};


uint16_t fp16_sin(uint16_t x) {
	uint16_t x2, sign, q, q1;
	
	sign = x & 0x8000;
	x = fp16_abs(x);

	//0.0209 rad, underflow
	if(fp16_lte(x, 0x255A)) 
	 return 0;
	
 x = fp16_sub(x,	fp16_mul(fp16_trunc(fp16_mul(0x3118, x)), 0x4648));
 q1 = (uint16_t) fp16_lrint(fp16_mul(0x3918, x));
 q = q1 + 1;
 
 x = fp16_sub(x, fp16_mul(0x3e48, fp16_tofloat16(q1)));

 sign = sign ^ (q > 2);
 
 //flip
 if(q == 2 || q == 4)
  x = fp16_sub(0x3E48, x);


	//0.0209 rad, underflow
	if(fp16_lte(x, 0x255A)) 
	 return 0;

	x2 = fp16_mul(x, x);
	return (sign ? 0x8000 : 0) | fp16_fma(fp16_mul(x, x2), fp16_fma(fp16_fma(c[0], x2, c[1]), x2, c[2]), x);
}

