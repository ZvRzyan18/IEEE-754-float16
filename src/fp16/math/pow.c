#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

#include <fenv.h>

static const fp8x23 c[16] = {
	0x39658674,
	0x3AA252D4,
	0x3C1EADBC,
	0x3D633F3A,
	0x3E75FEFF,
	0x3F317214,
	0x3F800000,

	0xBC14E9A5,
	0x3DFDA2D1,
	0xBF3F945A,
	0x4028E1AC,
	0xC0C06E0A,
	0x41149364,
	0xC1201D3A,
	0x41027A14,
	0xC05B39A4,
};


/*
 power function
 
 pow(x, y) = x ^ y
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


static inline fp8x23 __log2(fp8x23 x) {
 int32_t whole;
 fp8x23 m, poly;
 
 m = 1065353216U | (x & 0x007FFFFF);
 whole = (x >> 23) - 127;
	poly = fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[7], m), c[8]), m), c[9]), m), c[10]), m), c[11]), m), c[12]), m), c[13]), m), c[14]), m), c[15]);
 return fp32_add(fp32_longtofloat32(whole), poly);
}


/*
 bit useless since it easily reach overflow
*/
fp5x10 fp16_pow(fp5x10 x, fp5x10 y) {
	fp8x23 out, mx, is_negative, sign;
	fp5x10 bx, by, x_sign, y_sign;
	
	bx = x & 0x7FFF;
	by = y & 0x7FFF;
	x_sign = x & 0x8000;
	y_sign = y & 0x8000;

 /*
  special cases
 */
	if(by == 0)
	 return 0x3C00;
	if(by == 0x3C00)
	 return x;


	if(bx == 0 && (!y_sign && by > 0 && by <= 0x7C00))
	 return 0;
 if(bx == 0x7C00 && by > 0 && by <= 0x7C00 && !y_sign)
  return 0x7C00;
 if(bx == 0x7C00 && by > 0 && by <= 0x7C00 && y_sign)
  return 0;

 //either x or y is nan or inf
 if(bx >= 0x7C00 || by >= 0x7C00) {
	 if(bx > 0x7C00 || by > 0x7C00)
	  return 0x7C01;
 	if(bx > 0x3C00 && by == 0x7C00 && !y_sign)
	  return 0x7C00;
	 if(bx > 0x3C00 && by == 0x7C00 && y_sign)
	  return 0;
 	if(bx < 0x3C00 && by == 0x7C00 && !y_sign)
	  return 0;
 	if(bx < 0x3C00 && by == 0x7C00 && y_sign)
	  return 0x7C00;
 	if(bx == 0x3C00 && by == 0x7C00)
	  return 0x7C01;
	 if(bx == 0 && (!y_sign && by > 0x7C00))
	  return 0x7C00;
 }
 
	sign = (fp8x23)(x & 0x8000) << 16;
	if(sign) 
	 return 0x7C01;
	mx = __fp32_tofloat32(x);
	if((mx >> 23) < 127) {
	 mx = fp32_div(c[6], mx);
	 sign = 0x8000;
	}
 mx = fp32_mul(__log2(mx) | sign, __fp32_tofloat32(y));

	is_negative = mx & 0x80000000;
	
	if((mx & 0x7FFFFFFF) >= 0x7F800000) //inf, nan
	 return mx;
	
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

