#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 atan 4 degree polynomial
*/
static const uint32_t c[7] = {
	0xBC9A60D6,
	0x3D8F5DEB,
	0xBE0660F6,
	0x3E4B8783,
	0xBEAAA7D5,
	0x3fc90fdb, //pio2
	0x3F800000, //one
};


uint16_t fp16_atan(uint16_t x) {
	uint16_t sign;
	uint32_t mx, x2, x3, poly, high;
	
	//sign handle
	sign = x & 0x8000;

	x &= 0x7FFF;
	if(x == 0x7C00) //inf
	 return __fp32_tofloat16(c[5]) | sign; //pio2
	 
	if(x > 0x7C00) //nan
	 return 0x7C01; //nan

	mx = __fp32_tofloat32(x);
	
	high = mx > c[6];
	if(high) // |x| > 1.0
	 mx = fp32_div(c[6], mx);
	
	x2 =    fp32_mul(mx, mx);
	x3 =    fp32_mul(x2, mx);
 poly =  fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x2), c[1]), x2), c[2]), x2), c[3]), x2), c[4]);
	mx =    fp32_add(mx, fp32_mul(x3, poly));

 if(high) // |x| > 1.0
  mx = fp32_sub(c[5], mx);
 return __fp32_tofloat16(mx) | sign;
}
