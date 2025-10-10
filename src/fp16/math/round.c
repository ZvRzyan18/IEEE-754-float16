#include "fp16/math.h"
#include "fp16/float16.h"

/*
 round to nearest
*/
fp5x10 fp16_round(fp5x10 x) {
	fp5x10 x_bits, sign;

 x_bits = x;
 sign = x_bits & 0x8000;
	x_bits &= 0x7FFF;
 	
	//inf, nan
 if(x_bits >= 0x7C00)
  return x;
 	
 fp5x10 half = 0x3800; // 0.5f
 fp5x10 one = 0x3C00; // 1.0f
 
 fp5x10 whole = fp16_trunc(x_bits);
 fp5x10 frac = fp16_sub(x_bits, whole);
 
 if(fp16_gte(frac, half))
  return fp16_add(whole, one) | sign;

 return whole | sign;
}



