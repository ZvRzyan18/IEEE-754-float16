#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"


static const uint32_t c[8] = {
	0x3D9E9188,
	0x3CD61314,
	0x3D9DF4C6,
	0x3E2AA0DC,
	
	0x3F000000, //half
	0x3F800000, //one
	0x3fc90fdb, //pio2
	0x40000000, //two
};


static inline uint32_t __sqrt(uint32_t x) {
	uint32_t x_bits, mx, x_exponent;
	x_bits = x;
	x_exponent = ((x_bits >> 23) - 127) >> 1;
	x_bits = (x_bits & 0x007FFFFF) | ((x_exponent + 127) << 23);
	mx = x_bits;
	
	for(int i = 0; i < 3; i++) {
 	mx = fp32_mul(c[4], fp32_add(mx, fp32_div(x, mx)));
 	mx = fp32_mul(c[4], fp32_add(mx, fp32_div(x, mx)));
	}
 return	mx;
}


uint16_t fp16_acos(uint16_t x) {
	uint32_t mx, x2, x3, poly, high, sign;
	
	sign = (uint32_t)(x & 0x8000) << 16;
	x &= 0x7FFF;
	
	if(x > 0x3C00)
	 return 0x7C00; //inf
	
	mx = __fp32_tofloat32(x);
	
	high = mx >= c[4];
	if(high) 
	 mx = __sqrt(fp32_mul(fp32_sub(c[5], mx), c[4]));
	 
	x2 =    fp32_mul(mx, mx);
	x3 =    fp32_mul(x2, mx);
 poly =  fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x2), c[1]), x2), c[2]), x2), c[3]);
	mx =    fp32_add(mx, fp32_mul(x3, poly));
	
 if(high)
 	mx = fp32_sub(c[6], fp32_mul(c[7], mx));
 
 return __fp32_tofloat16(fp32_sub(c[6], mx | sign));
}

