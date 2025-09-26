#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"


static const uint32_t c[16] = {
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
static inline uint32_t __exp2(uint32_t x, uint32_t *out) {
	uint32_t whole_part, poly;
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


static inline uint32_t __log2(uint32_t x) {
 int32_t whole;
 uint32_t m, poly;
 
 m = 1065353216U | (x & 0x007FFFFF);
 whole = (x >> 23) - 127;
	poly = fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[7], m), c[8]), m), c[9]), m), c[10]), m), c[11]), m), c[12]), m), c[13]), m), c[14]), m), c[15]);
 return fp32_add(fp32_longtofloat32(whole), poly);
}

//TODO : handle inf, nan correctly
/*
 bit useless since it easily reach overflow
*/
uint16_t fp16_pow(uint16_t x, uint16_t y) {
	uint32_t out, mx, is_negative, sign;
	
	sign = (uint32_t)(x & 0x8000) << 16;
	if(sign) 
	 return 0x7C01;
	if((x & 0x7FFF) >= 0x7C00) //inf, nan
	 return x;
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
		 return 0x7C00;
		break;
		case 2: //underflow
		 return 0;
		break;
	}
	return 0x7C01; //unreachable
}

