#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

#include <fenv.h>

static const fp8x23 c[8] = {
	0x39658674,
	0x3AA252D4,
	0x3C1EADBC,
	0x3D633F3A,
	0x3E75FEFF,
	0x3F317214,
	0x3F800000,
	0x3FB8AA3B, //log2(e)
};

/*
 exponential
 
 exp(x) = e ^ x
*/
static inline fp8x23 __exp2(fp8x23 x, fp8x23 *out) {
	fp8x23 whole_part, poly;
	int32_t whole;
	
	whole = (int32_t)fp32_floattolong(x);
	
	whole_part = (whole + 127) << 23;
	if(whole > 15)
	 return 1;
	if(whole < -14)
	 return 2;
	x = fp32_sub(x, fp32_longtofloat32(whole));
	poly = fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x), c[1]), x), c[2]), x), c[3]), x), c[4]), x), c[5]), x), c[6]);
	*out = fp32_mul(whole_part, poly);
	return 0;
}


fp5x10 fp16_exp(fp5x10 x) {
	fp8x23 out, mx, is_negative;
	
	is_negative = x & 0x8000;
	
	if((x & 0x7FFF) >= 0x7C00) //inf, nan
	 return x;
	
	mx = __fp32_tofloat32(x & 0x7FFF);
 mx = fp32_mul(mx, c[7]);
	switch(__exp2(mx, &out)) {
		case 0:
			if(is_negative)
	   return __fp32_tofloat16(fp32_div(c[6], out));
   else
		 return __fp32_tofloat16(out);
		break;
		case 1: //overflow
	 	feraiseexcept(FE_OVERFLOW);
		 return 0x7C00;
		break;
		case 2: //underflow
		 feraiseexcept(FE_UNDERFLOW);
		 return 0;
		break;
	}
	return 0x7C01; //unreachable
}

