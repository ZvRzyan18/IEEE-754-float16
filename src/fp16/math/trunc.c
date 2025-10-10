#include "fp16/math.h"
#include "fp16/float16.h"

/*
 round toward zero
*/
fp5x10 fp16_trunc(fp5x10 x) {
 fp5x10 x_bits, out_bits, sign, mantissa;

 x_bits = x;
	sign = x_bits & 0x8000;
 x_bits &= 0x7FFF;
 	
 //inf, nan
 if(x_bits >= 0x7C00)
  return x;
 	
	int16_t exponent = (x_bits >> 10) - 15;
 	
 if(exponent < 0)
 	return 0;
 	 
 mantissa = x_bits & 0x03FF;
 	
 if(exponent >= 10)
  return x; //integral
 	
	fp5x10 mask = 0xFFFF << (10 - exponent);
 mantissa &= mask;
 out_bits = sign | ((exponent + 15) << 10) | mantissa;
 return out_bits;
}

