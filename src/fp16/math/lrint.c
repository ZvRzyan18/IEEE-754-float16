#include "fp16/math.h"
#include "fp16/float16.h"

#include <fenv.h>

/*
 round based on the current rounding mode
 rint()/lrint() functions
*/

//trunc
static inline long irint_z(uint16_t x) {
	uint16_t x_bits, mantissa, sign;
 long integer_part;
 
 x_bits = x;
 	
 sign = x_bits & 0x8000;
	x_bits &= 0x7FFF;
 	
	//inf, nan
	if(x_bits >= 0x7C00)
  return 0x7FFFFFFF;
 	
 int16_t exponent = (x_bits >> 10) - 15;
 	
 if(exponent < 0)
 	return 0;
 	 
 mantissa = x_bits & 0x03FF;
 	
 mantissa |= (1 << 10);
 	
 if(exponent >= 10)
  integer_part = mantissa << (exponent - 10);
 else
  integer_part = mantissa >> (10 - exponent);
 return sign ? -integer_part : integer_part;
}
 
 
//ceil
static inline long irint_p(uint16_t x) {
	uint16_t x_bits, sign, mantissa;
 long integer_part;

 x_bits = x; 	
 sign = x_bits & 0x8000;
 x_bits &= 0x7FFF;
 	
 //inf, nan
 if(x_bits >= 0x7C00)
 	return 0x7FFFFFFF;
 	 
 	int16_t exponent = (x_bits >> 10) - 15;
 	
 if(exponent < 0)
 	return 0;
 	 
	mantissa = x_bits & 0x03FF;
 	
 mantissa |= (1 << 10);
 	
 if(exponent >= 10)
  integer_part = mantissa << (exponent - 10);
 else
  integer_part = mantissa >> (10 - exponent);
  
 if(sign == 0) {
  int is_integral = (exponent < 10) && ((mantissa & ((1 << (10 - exponent)) - 1)) != 0);
  integer_part = integer_part + (is_integral ? 1 : 0);
 }
 return integer_part;
}
 
 
 
//floor
static inline long irint_n(uint16_t x) {
 uint16_t x_bits, sign, mantissa;
	long integer_part;
	
 x_bits = x;
 sign = x_bits & 0x8000;
	x_bits &= 0x7FFF;
 	
 //inf, nan
 if(x_bits >= 0x7C00)
 	return 0x7FFFFFFF;
 	 
 int16_t exponent = (x_bits >> 10) - 15;
 	
	if(exponent < 0)
 	return 0;
 	 
 mantissa = x_bits & 0x03FF;
 	
	mantissa |= (1 << 10);
 	
	if(exponent >= 10)
  integer_part = mantissa << (exponent - 10);
 else
  integer_part = mantissa >> (10 - exponent);
  
 if(sign != 0) {
  int is_integral = (exponent < 10) && ((mantissa & ((1 << (10 - exponent)) - 1)) != 0);
  integer_part = -integer_part - (is_integral ? 1 : 0);
 }
 return integer_part;
}
 
 
//round
static inline long irint_r(uint16_t x) {
	uint16_t x_bits, sign, mantissa;
 long integer_part;

 x_bits = x;
 sign = x_bits & 0x8000;
	x_bits &= 0x7FFF;
 	
	//inf, nan
	if(x_bits >= 0x7C00)
  return 0x7FFFFFFF;
 	
 int16_t exponent = (x_bits >> 10) - 15;
 	
 if(exponent < 0)
  return 0;
 	 
	mantissa = x_bits & 0x03FF;
 mantissa |= (1 << 10);
 	
 if(exponent >= 10)
   integer_part = mantissa << (exponent - 10);
 else {
  int shift = 10 - exponent;
  uint16_t roundBit = (mantissa >> (shift - 1)) & 1;
  integer_part = mantissa >> shift;
  integer_part += roundBit;
 }
 return sign ? -integer_part : integer_part;
}
 

long fp16_lrint(uint16_t x) {
 switch(fegetround()) {
 	case FE_TOWARDZERO:
 	 return irint_z(x);
 	break;
 	case FE_DOWNWARD:
 	 return irint_n(x);
 	break;
 	case FE_UPWARD:
 	 return irint_p(x);
 	break;
 	case FE_TONEAREST:
 	 return irint_r(x);
 	break;
 }
 return irint_z(x);
}
