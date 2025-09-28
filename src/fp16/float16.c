#include "fp16/float16.h"
#include "fp16/float32.h"

#include <fenv.h>
/*
 IEEE 754 float 16 
*/

//bias : 15
//significand bit : 10
//max + exp : 15
//min - exp : -14

typedef union {
	float f;
	uint32_t i;
} __fp32_bits;



uint16_t fp16_tofloat16(float x) {
 __fp32_bits bits;
 bits.f = x;
 //for implementation of __fp32_tofloat16 see @float32.h 
 return __fp32_tofloat16(bits.i);
}



float fp16_tofloat32(uint16_t x) {
 __fp32_bits bits;
 //for implementation of __fp32_tofloat32 see @float32.h 
 bits.i = __fp32_tofloat32(x);
 return bits.f;
}



uint16_t fp16_longtofloat16(int64_t x) {
 uint16_t sign, input, exponent, mantissa;
 int16_t msb;
 
 sign = ((x < 0) ? 1 : 0) << 15;
 input = (sign != 0) ? -x : x;
 msb = 15;

 while(msb >= 0 && ((input >> msb) & 1) == 0)
  --msb;
 
 exponent = (msb + 15) << 10;
 mantissa = 0;
 
 if(msb > 0) {
  int shift = msb - 10;
  if(shift >= 0)
   mantissa = (input >> shift) & 0x03FF;
   else
  mantissa = (input << -shift) & 0x03FF;
 }
 return sign | exponent | mantissa;
}


/*
 compare operator
*/

uint32_t fp16_gt(uint16_t a, uint16_t b) {
	uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
	a &= 0x7FFF;
 b &= 0x7FFF;
 if(a <= 0x7C00 && b <= 0x7C00) {
  if(a_sign && !b_sign)
 	 return 0;
 	if(!a_sign && b_sign)
   return 1;
 	if(a_sign && b_sign)
   return a < b;
 	if(!a_sign && !b_sign)
  return a > b;
 }
 //nan
	return 0;
}
 
 
uint32_t fp16_lt(uint16_t a, uint16_t b) {
 uint16_t a_sign = a & 0x8000;
	uint16_t b_sign = b & 0x8000;
	a &= 0x7FFF;
 b &= 0x7FFF;
 if(a <= 0x7C00 && b <= 0x7C00) {
 	if(a_sign && !b_sign)
   return 1;
  if(!a_sign && b_sign)
 	 return 0;
  if(a_sign && b_sign)
   return a > b;
  if(!a_sign && !b_sign)
  	return a < b;
 }
 //nan
	return 0;
}
 
uint32_t fp16_gte(uint16_t a, uint16_t b) {
 uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
	a &= 0x7FFF;
	b &= 0x7FFF;
	if(a <= 0x7C00 && b <= 0x7C00) {
 	if(a_sign && !b_sign)
 	 return 0;
 	if(!a_sign && b_sign)
   return 1;
 	if(a_sign && b_sign)
  	return a <= b;
 	if(!a_sign && !b_sign)
   return a >= b;
 }
	//nan
 return 0;
}
 
 
uint32_t fp16_lte(uint16_t a, uint16_t b) {
	uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
 a &= 0x7FFF;
	b &= 0x7FFF;
 if(a <= 0x7C00 && b <= 0x7C00) {
 	if(a_sign && !b_sign)
 	 return 1;
 	if(!a_sign && b_sign)
   return 0;
  if(a_sign && b_sign)
   return a >= b;
 	if(!a_sign && !b_sign)
   return a <= b;
 }
 //nan
	return 0;
}
 
 
uint32_t fp16_eq(uint16_t a, uint16_t b) {
	uint16_t sign_a = a & 0x8000;
	uint16_t sign_b = b & 0x8000;
	a &= 0x7FFF;
 b &= 0x7FFF;
	if(a <= 0x7C00 && b <= 0x7C00) {
 	return a == b && (sign_a == sign_b);
	}
 //nan
	return 0;
}


uint32_t fp16_neq(uint16_t a, uint16_t b) {
 return !fp16_eq(a, b);
}


/*
 arithmetic operator
*/


static inline uint16_t unsigned_add_bit(uint16_t a, uint16_t b, uint16_t *grs) {
 uint16_t a_bits, b_bits, out_bits, final_exponent, final_mantissa, shift, inexact, grs_count;
 a_bits = a;
 b_bits = b;
	
	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
 int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;
 
 uint16_t a_mantissa = (a_bits & 0x03FF);
 uint16_t b_mantissa = (b_bits & 0x03FF);

 // add leading ones
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (uint16_t)((b_mantissa >> (shift-3)) & 0x0007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else if (a_exponent < b_exponent) {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (uint16_t)((a_mantissa >> (shift-3)) & 0x0007);
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 } else {
 	(*grs) = 0;
 	final_exponent = a_exponent;
 }
 
 final_mantissa = a_mantissa + b_mantissa;
 
 grs_count = 0;
 //normalize
 while(final_mantissa >= (1 << 11)) {
  if(grs_count == 0)
   (*grs) = 0;
  if(grs_count < 3)
   (*grs) |= (final_mantissa & 1) << (2 - grs_count);
  inexact |= (final_mantissa & 1);
	 final_mantissa >>= 1;
		final_exponent++;
		grs_count++;
 }
 
 if(inexact)
  feraiseexcept(FE_INEXACT);

 out_bits = 0;
 out_bits |= ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}



static inline uint16_t unsigned_sub_bit(uint16_t a, uint16_t b, uint16_t *grs) {
 uint16_t a_bits, b_bits, out_bits, shift, inexact;
 int16_t final_mantissa, final_exponent;
 a_bits = a;
	b_bits = b;
	
	if(a_bits == b_bits)	{
	 out_bits = 0;
 	return out_bits;
 }

	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
	int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;

 uint16_t a_mantissa = (a_bits & 0x03FF);
 uint16_t b_mantissa = (b_bits & 0x03FF);

 // add leading ones
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (uint16_t)((b_mantissa >> (shift-3)) & 0x0007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (uint16_t)((a_mantissa >> (shift-3)) & 0x0007);
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 }

 final_mantissa = a_mantissa - b_mantissa;
 
 //normalize
 while((final_mantissa & (1 << 10)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 if(inexact)
  feraiseexcept(FE_INEXACT);

 out_bits = ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}



uint16_t fp16_add(uint16_t a, uint16_t b) {
 uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
 uint16_t grs, sum;
  
 a &= 0x7FFF;
 b &= 0x7FFF;
  
 //inf nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  return 0x7C01;
 }
 
 grs = 0;
 sum = 0;
 if(a_sign == b_sign) {
  sum = unsigned_add_bit(a, b, &grs) | a_sign;
 } else {
 	if(a > b)	{
   sum = unsigned_sub_bit(a, b, &grs) | a_sign;
  } else	{
  	sum = unsigned_sub_bit(b, a, &grs) | b_sign;
 	}
 }

 /*
  inexact rounding
 */
  uint16_t guard = (grs >> 2) & 1;
  uint16_t round = (grs >> 1) & 1;
  uint16_t sticky = grs & 1;

  uint16_t out_mantissa = sum & 0x03FF;
  uint16_t exponent = (sum & 0x7FFF) >> 10;
  uint16_t sign = sum & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
   out_mantissa |= (1 << 10);
  
  uint16_t increment = 0;

  switch(fegetround()) {
   case FE_TONEAREST:
   if(guard && (round || sticky || (out_mantissa & 1)))
    increment = 1;
   break;
   case FE_TOWARDZERO:
    increment = 0;
   break;
   case FE_UPWARD:
    if(sign) 
     increment = 0;
    else 
     increment = guard || round || sticky;
   break;
   case FE_DOWNWARD:
    if(sign)
     increment = 1;
    else 
     increment = 0;
   break;
   default: //to nearest, default
    if(guard && (round || sticky || (out_mantissa & 1)))
     increment = 1;
   break;
  }

 if(increment) {
  out_mantissa++;
  //overflow mantissa, adjust exponent and re-normalize
  if(out_mantissa >= (1 << 11)) {
   out_mantissa >>= 1;
   exponent++;
   
   //too large, overflow exponent
   if(exponent > 30) {
   	feraiseexcept(FE_OVERFLOW);
    return 0x7C00;
   }
   
  }
 }
 return sign | (exponent << 10) | (out_mantissa & 0x03FF);
}


uint16_t fp16_sub(uint16_t a, uint16_t b) {
 uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
  
 uint16_t grs, diff;
 
 grs = 0;
 diff = 0;
   
 a &= 0x7FFF;
 b &= 0x7FFF;

 //inf nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  return 0x7C01;
 }

 if(a_sign == b_sign) {
  if(a > b)
  	diff = unsigned_sub_bit(a, b, &grs) | a_sign;
  else if(b > a)
  	diff = unsigned_sub_bit(b, a, &grs) | (!a_sign ? 0x8000 : 0);
  else
   diff = 0;
 } else {
 	 diff = unsigned_add_bit(a, b, &grs) | a_sign;
 }

 /*
  inexact rounding
 */
  uint16_t guard = (grs >> 2) & 1;
  uint16_t round = (grs >> 1) & 1;
  uint16_t sticky = grs & 1;

  uint16_t out_mantissa = diff & 0x03FF;
  uint16_t exponent = (diff & 0x7FFF) >> 10;
  uint16_t sign = diff & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
   out_mantissa |= (1 << 10);
  
  uint16_t increment = 0;

  switch(fegetround()) {
   case FE_TONEAREST:
   if(guard && (round || sticky || (out_mantissa & 1)))
    increment = 1;
   break;
   case FE_TOWARDZERO:
    increment = 0;
   break;
   case FE_UPWARD:
    if(sign) 
     increment = 0;
    else 
     increment = guard || round || sticky;
   break;
   case FE_DOWNWARD:
    if(sign)
     increment = 1;
    else 
     increment = 0;
   break;
   default: //to nearest, default
    if(guard && (round || sticky || (out_mantissa & 1)))
     increment = 1;
   break;
  }

 if(increment) {
  out_mantissa++;
  //overflow mantissa, adjust exponent and re-normalize
  if(out_mantissa >= (1 << 11)) {
   out_mantissa >>= 1;
   exponent++;
   
   //too large, overflow exponent
   if(exponent > 30) {
   	feraiseexcept(FE_OVERFLOW);
    return 0x7C00;
   }
   
  }
 }
 return sign | (exponent << 10) | (out_mantissa & 0x03FF);
}


uint16_t fp16_mul(uint16_t a, uint16_t b) {
	uint16_t a_bits, b_bits, out_bits, sign, inexact, grs, grs_count;
	int16_t exponent;
	uint32_t mantissa;
	
 a_bits = a;
	b_bits = b;
	
 //sign bit
 // +, + = +
 // -, - = +
 // +, - = -
 // -, + = -
	sign = ((a_bits & 0x8000) ^ (b_bits & 0x8000));

 a_bits &= 0x7FFF;
	b_bits &= 0x7FFF;
	
	if(a_bits == 0 || b_bits == 0)
	 return 0;

 if(b_bits == 0x3C00)
  return a ^ (b_bits & 0x8000);
	if(a_bits == 0x3C00)
 	return b ^ (a_bits & 0x8000);
 	
 //inf nan
 if(a_bits >= 0x7C00 || b_bits >= 0x7C00) {
 	return 0x7C01;
 }
	
	int16_t a_exponent = (int16_t)((a_bits) >> 10) - 15;
	int16_t b_exponent = (int16_t)((b_bits) >> 10) - 15;

 exponent = a_exponent + b_exponent;
 
 uint16_t a_mantissa = a_bits & 0x03FF;
 uint16_t b_mantissa = b_bits & 0x03FF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 //multiply and round the low mantissa
 mantissa = (uint32_t)a_mantissa * (uint32_t)b_mantissa;
 grs = (uint16_t)((mantissa >> 7) & 0x00000007);
 inexact = (mantissa & 0x000003FF) != 0;
	mantissa >>= 10;
	
	grs_count = 0;
 //normalize
 while(mantissa >= (1 << 11)) {
  if(grs_count == 0)
   grs = 0;
  if(grs_count < 3)
   grs |= (mantissa & 1) << (2 - grs_count);
 	inexact |= mantissa & 1;
		mantissa >>= 1;
		exponent++;
		grs_count++;
 }
 
 if(inexact)
  feraiseexcept(FE_INEXACT);

 out_bits = sign | ((exponent + 15) << 10) | ((uint16_t)mantissa & 0x03FF);

 if(exponent > 15) {
 	feraiseexcept(FE_OVERFLOW);
 	return 0x7C00; //overflow
 } if(exponent < -14) {
  mantissa |= 1 << 10;
  int shift = -14 - exponent;
  if(shift > 10 && !(shift < 0)) {
   //undeflow
   feraiseexcept(FE_UNDERFLOW);
   return sign;
  } else {
   //subnormal
   mantissa >>= shift;
   out_bits = sign | ((uint16_t)mantissa & 0x03FF);
  }
 }
 
 
 /*
  inexact rounding
 */
  uint16_t guard = (grs >> 2) & 1;
  uint16_t round = (grs >> 1) & 1;
  uint16_t sticky = grs & 1;

  uint16_t out_mantissa = out_bits & 0x03FF;
  exponent = (out_bits & 0x7FFF) >> 10;
  sign = out_bits & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
   out_mantissa |= (1 << 10);
  
  uint16_t increment = 0;

  switch(fegetround()) {
   case FE_TONEAREST:
   if(guard && (round || sticky || (out_mantissa & 1)))
    increment = 1;
   break;
   case FE_TOWARDZERO:
    increment = 0;
   break;
   case FE_UPWARD:
    if(sign) 
     increment = 0;
    else 
     increment = guard || round || sticky;
   break;
   case FE_DOWNWARD:
    if(sign)
     increment = 1;
    else 
     increment = 0;
   break;
   default: //to nearest, default
    if(guard && (round || sticky || (out_mantissa & 1)))
     increment = 1;
   break;
  }

 if(increment) {
  out_mantissa++;
  //overflow mantissa, adjust exponent and re-normalize
  if(out_mantissa >= (1 << 11)) {
   out_mantissa >>= 1;
   exponent++;
   
   //too large, overflow exponent
   if(exponent > 30) {
   	feraiseexcept(FE_OVERFLOW);
    return 0x7C00;
   }
   
  }
 }
 return sign | (exponent << 10) | (out_mantissa & 0x03FF);
}
 

uint16_t fp16_div(uint16_t a, uint16_t b) {
	uint16_t a_bits, b_bits, out_bits, sign, inexact, grs;
	int16_t exponent;
	uint32_t mantissa;
	
	a_bits = a;
	b_bits = b;

 //sign bit
 // +, + = +
 // -, - = +
 // +, - = -
 // -, + = -
 sign = ((a_bits & 0x8000) ^ (b_bits & 0x8000));

	a_bits &= 0x7FFF;
 b_bits &= 0x7FFF;
 	
 if(b_bits == 0x3C00)
 	return a_bits | sign;

	if(b_bits == 0) {
	 feraiseexcept(FE_DIVBYZERO);
	 return 0x7C00 | sign; //infinity
	}
	if(a_bits == 0)
 	return 0;

 //inf nan
 if(a_bits >= 0x7C00 || b_bits >= 0x7C00) {
 	return 0x7C01;
 }
	
	int16_t a_exponent = (int16_t)((a_bits) >> 10) - 15;
	int16_t b_exponent = (int16_t)((b_bits) >> 10) - 15;

 exponent = a_exponent - b_exponent;
 
 uint16_t a_mantissa = a_bits & 0x03FF;
 uint16_t b_mantissa = b_bits & 0x03FF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 //divide mantissa
 mantissa = (uint32_t)(a_mantissa);
 mantissa <<= 10;
 mantissa /= (uint32_t)b_mantissa;

 //shift to left more so we can catch the lost bits during division
 inexact = (uint16_t)( ((((uint32_t)(a_mantissa) << 20) / (((uint32_t)b_mantissa) << 10)) & 0x000003FF) );
 grs = (inexact >> 7) & 0x0007;

 //normalize
 while((mantissa & (1 << 10)) == 0 && mantissa != 0) {
  mantissa <<= 1;
  exponent--;
 }
 
 if(inexact)
  feraiseexcept(FE_INEXACT);
 
 if(exponent > 15) {
 	feraiseexcept(FE_OVERFLOW);
 	return 0x7C00; //overflow
 } if(exponent < -14) {
  mantissa |= 1 << 10;
  int shift = -14 - exponent;
  if(shift > 10 && !(shift < 0)) {
   //undeflow
   feraiseexcept(FE_UNDERFLOW);
   return sign;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((uint16_t)mantissa & 0x03FF);
  }
 }
	out_bits = sign | ((exponent + 15) << 10) | ((uint16_t)mantissa & 0x03FF);

 /*
  inexact rounding
 */
  uint16_t guard = (grs >> 2) & 1;
  uint16_t round = (grs >> 1) & 1;
  uint16_t sticky = grs & 1;

  uint16_t out_mantissa = out_bits & 0x03FF;
  exponent = (out_bits & 0x7FFF) >> 10;
  sign = out_bits & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
   out_mantissa |= (1 << 10);
  
  uint16_t increment = 0;

  switch(fegetround()) {
   case FE_TONEAREST:
   if(guard && (round || sticky || (out_mantissa & 1)))
    increment = 1;
   break;
   case FE_TOWARDZERO:
    increment = 0;
   break;
   case FE_UPWARD:
    if(sign) 
     increment = 0;
    else 
     increment = guard || round || sticky;
   break;
   case FE_DOWNWARD:
    if(sign)
     increment = 1;
    else 
     increment = 0;
   break;
   default: //to nearest, default
    if(guard && (round || sticky || (out_mantissa & 1)))
     increment = 1;
   break;
  }

 if(increment) {
  out_mantissa++;
  //overflow mantissa, adjust exponent and re-normalize
  if(out_mantissa >= (1 << 11)) {
   out_mantissa >>= 1;
   exponent++;
   
   //too large, overflow exponent
   if(exponent > 30) {
   	feraiseexcept(FE_OVERFLOW);
    return 0x7C00;
   }
   
  }
 }
 return sign | (exponent << 10) | (out_mantissa & 0x03FF);
}



 
