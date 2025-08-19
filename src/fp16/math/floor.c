#include "fp16/math.h"
#include "fp16/float16.h"

uint16_t fp16_floor(uint16_t x) {
	uint16_t x_bits, out_bits, sign, mantissa, mask;
  	
 x_bits = x;
 sign = x_bits & 0x8000;
	x_bits &= 0x7FFF;
 	
	//inf, nan
	if(x_bits >= 0x7C00)	{
	 out_bits = 0x7C01;
  return out_bits;
	}
 	
	int16_t exponent = (x_bits >> 10) - 15;
 	
 if(exponent < 0)
  return sign ? 0xBC00 : 0;
 	 
 mantissa = x_bits & 0x03FF;
 	
	if(exponent >= 10)
  return x; //integral
 
 int has_fraction = ((mantissa & ((1 << (10 - exponent)) - 1)) != 0);

 mask = 0xFFFF << (10 - exponent);
 mantissa &= mask;
 out_bits = sign | ((exponent + 15) << 10) | mantissa;
 if(sign && has_fraction)
  out_bits = fp16_add(out_bits, 0xBC00);
 return out_bits;
}

