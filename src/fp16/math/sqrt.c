#include "fp16/math.h"
#include "fp16/float16.h"

#include <fenv.h>
/*
 square root
 
 sqrt(x) = x ^ 1/2
*/
fp5x10 fp16_sqrt(fp5x10 x) {
 //Quake III fast inverse sqrt
 fp5x10 x_bits;

 x_bits = x;
 if(x_bits & 0x8000) {
  feraiseexcept(FE_INVALID);
  return 0x7C01;
 }

 //inf, nan
	if((x_bits & 0x7FFF) >= 0x7C00)
 	return x;

 fp5x10 x_half = fp16_mul(x, 0x3800);
 fp5x10 three_half = 0x3E00;
 	
 x_bits = 0x59BA - (x_bits >> 1);
 x_bits = fp16_mul(x_bits, fp16_sub(three_half, fp16_mul(x_half, fp16_mul(x_bits, x_bits))));
 x_bits = fp16_mul(x_bits, fp16_sub(three_half, fp16_mul(x_half, fp16_mul(x_bits, x_bits))));
 return fp16_mul(x, x_bits);
}

