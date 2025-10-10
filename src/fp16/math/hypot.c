#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 hypothenouse
*/

/*
 hypot
          • (a, b)
         /|
        / |
 c^2 ->/  |
      /   |
     /    |
    /     |
   •-------
   ^
  origin
 
 pythagorean theorem
 c^2 = a^2 + b^2
 
 to get the exact hypothenouse value, the exponent needs to get rid off
 
 c = sqrt(c^2)
*/

//32 bit sqrt
static inline fp8x23 __sqrt(fp8x23 x) {
	fp8x23 x_bits, mx;
	int32_t x_exponent;
	x_bits = x;
	x_exponent = (int32_t)((x_bits >> 23) - 127) >> 1;
	x_bits = (x_bits & 0x007FFFFF) | ((fp8x23)(x_exponent + 127) << 23);
	mx = x_bits;
	
	for(int i = 0; i < 3; i++) {
 	mx = fp32_mul(0x3F000000, fp32_add(mx, fp32_div(x, mx)));
 	mx = fp32_mul(0x3F000000, fp32_add(mx, fp32_div(x, mx)));
	}
 return	mx;
}


fp5x10 fp16_hypot(fp5x10 x, fp5x10 y) {
	fp8x23 mx, my;
	
	mx = __fp32_tofloat32(x);
	my = __fp32_tofloat32(y);
	mx = fp32_mul(mx, mx);
	my = fp32_mul(my, my);
	
	return __fp32_tofloat16(__sqrt(fp32_add(mx, my)));
}

