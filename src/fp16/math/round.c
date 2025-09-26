#include "fp16/math.h"
#include "fp16/float16.h"

/*
 round to nearest
*/
uint16_t fp16_round(uint16_t x) {
	uint16_t x_bits, out_bits, sign;

 x_bits = x;
 sign = x_bits & 0x8000;
	x_bits &= 0x7FFF;
 	
	//inf, nan
 if(x_bits >= 0x7C00) {
  out_bits = 0x7C01;
  return out_bits;
 }
 	
 uint16_t half = 0x3800; // 0.5f
 uint16_t one = 0x3C00; // 1.0f
 
 uint16_t whole = fp16_trunc(x_bits);
 uint16_t frac = fp16_sub(x_bits, whole);
 
 if(fp16_gte(frac, half))
  return fp16_add(whole, one) | sign;

 return whole | sign;
}



