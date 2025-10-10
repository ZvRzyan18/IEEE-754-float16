#ifndef FP16_FLOAT32_H
#define FP16_FLOAT32_H

#include "fp16/fptypes.h"

#include <fenv.h>
/*
 IEEE 754 float 32 
 this is only here to extend the precision of half float
*/
//bias : 127
//significand bit : 23
//max + exp : 127
//min - exp : -126


static inline fp8x23 __fp32_tofloat32(fp5x10 x) {
 fp8x23 sign = x >> 15;
 fp8x23 exp16 = (x & 0x7FFF) >> 10;
 fp8x23 mant16 = x & 0x03FF;

 fp8x23 sign32 = sign << 31;
 fp8x23 exp32, mant32;

 if(exp16 == 0) {
  if(mant16 == 0) {
   return 0;
  } else {
   // subnormal
   int shift = 0;
   fp8x23 mant = mant16;
   while((mant & 0x400) == 0) {
    mant <<= 1;
    shift++;
   }
   mant &= 0x03FF;
   exp32 = -14 - shift + 127;
   mant32 = mant << 13;
  }
 } else if(exp16 == 0x1F) {
  //inf, nan
  exp32 = 0xFF;
  mant32 = mant16 ? (mant16 << 13) | 1 : 0;
 } else {
  exp32 = exp16 - 15 + 127;
  mant32 = mant16 << 13;
 }
 
 //no need rounding since no bits lost
 return sign32 | (exp32 << 23) | (mant32 & 0x007FFFFF);
}


static inline fp5x10 __fp32_tofloat16(fp8x23 x) {
 fp8x23 bits = x;
 fp8x23 sign = bits >> 31;
 int32_t exp32 = (bits & 0x7FFFFFFF) >> 23;
 fp8x23 mant32 = bits & 0x007FFFFF;
 
 fp5x10 grs = 0;
 fp5x10 inexact = 0;

 fp5x10 sign16 = sign << 15;
 fp5x10 exp16 = 0;
 fp5x10 mant16 = 0;
 
 if((x & 0x7FFFFFFF) == 0)
  return 0;

 if(exp32 == 0xFF) {
  //inf, nan
  exp16 = 0x1F;
  mant16 = mant32 ? 1 : 0;
 } else if(exp32 > 142) {
  //overflow
  feraiseexcept(FE_OVERFLOW);
  return 0x7C00;
 } else if(exp32 < 113) {
  if(exp32 < 103) {
   // too small for subnormal, round to zero
   feraiseexcept(FE_UNDERFLOW);
   return 0;
  } else {
   //subnormal
   fp8x23 mant = mant32 | 0x800000;
   int32_t shift = 113 - exp32;
   mant16 = mant >> (shift + 13);
   inexact |= (fp5x10)(mant & ((1 << (shift + 14))-1)) != 0;
   grs = (fp5x10)((mant >> (shift + 10)) & 0x00000007);
  }
 } else {
  exp16 = exp32 - 127 + 15;
  mant16 = mant32 >> 13;
  inexact |= (fp5x10)(mant32 & ((1 << 14)-1)) != 0;
  grs = (fp5x10)((mant32 >> 10) & 0x00000007);
 }
 
 if(inexact)
 	feraiseexcept(FE_INEXACT);
 else
 	return sign16 | (exp16 << 10) | mant16; //exact, no need rounding
 
 /*
  inexact rounding
 */
  fp5x10 guard = (grs >> 2) & 1;
  fp5x10 round = (grs >> 1) & 1;
  fp5x10 sticky = grs & 1;

  fp5x10 out_mantissa = mant16;
  
  //only add leading one for non subnormal
  if(exp16 > 0)
   out_mantissa |= (1 << 10);
  
  fp5x10 increment = 0;

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
   exp16++;
   
   //too large, overflow exponent
   if(exp16 > 30) {
   	feraiseexcept(FE_OVERFLOW);
    return 0x7C00;
   }
   
  }
 }
 mant16 = out_mantissa & 0x03FF;
 return sign16 | (exp16 << 10) | mant16;
}



static inline fp8x23 __unsigned_add_bit(fp8x23 a, fp8x23 b) {
 fp8x23 a_bits, b_bits, out_bits, final_exponent, final_mantissa, shift, inexact;
 a_bits = a;
 b_bits = b;
	
	int32_t a_exponent = (int32_t)(a_bits >> 23) - 127;
 int32_t b_exponent = (int32_t)(b_bits >> 23) - 127;
 
 fp8x23 a_mantissa = (a_bits & 0x007FFFFF);
 fp8x23 b_mantissa = (b_bits & 0x007FFFFF);

 // add leading ones
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else if (a_exponent < b_exponent) {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 } else {
 	final_exponent = a_exponent;
 }
 
 final_mantissa = a_mantissa + b_mantissa;
 
 inexact = 0;
 //normalize
 while(final_mantissa >= (1 << 24)) {
  inexact |= final_mantissa & 1;
	 final_mantissa >>= 1;
		final_exponent++;
 }
 
 if(inexact)
  feraiseexcept(FE_INEXACT);

 out_bits = 0;
 out_bits |= ((final_exponent + 127) << 23) | (final_mantissa & 0x007FFFFF);
 return out_bits;
}



static inline fp8x23 __unsigned_sub_bit(fp8x23 a, fp8x23 b) {
 fp8x23 a_bits, b_bits, out_bits, shift, inexact;
 int32_t final_mantissa, final_exponent;
 a_bits = a;
	b_bits = b;
	
	if(a_bits == b_bits)	{
	 out_bits = 0;
 	return out_bits;
 }

	int32_t a_exponent = (int32_t)(a_bits >> 23) - 127;
	int32_t b_exponent = (int32_t)(b_bits >> 23) - 127;

 fp8x23 a_mantissa = (a_bits & 0x007FFFFF);
 fp8x23 b_mantissa = (b_bits & 0x007FFFFF);
 
 // add leading ones
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 }

 final_mantissa = a_mantissa - b_mantissa;
 
 //normalize
 while((final_mantissa & (1 << 23)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 if(inexact)
  feraiseexcept(FE_INEXACT);

 out_bits = ((final_exponent + 127) << 23) | (final_mantissa & 0x007FFFFF);
 return out_bits;
}


static inline fp8x23 fp32_add(fp8x23 a, fp8x23 b) {
 fp8x23 a_sign = a & 0x80000000;
 fp8x23 b_sign = b & 0x80000000;
  
 a &= 0x7FFFFFFF;
 b &= 0x7FFFFFFF;
 
 //inf nan
 if(a >= 0x7F800000 || b >= 0x7F800000) {
  if(a == 0x7F800000 || b == 0x7F800000)
   return 0x7F800000;
  else
   return 0x7F800001;
 }
 
 if(a_sign == b_sign) {
  return __unsigned_add_bit(a, b) | a_sign;
 } else {
 	if(a > b)	{
   return __unsigned_sub_bit(a, b) | a_sign;
  } else	{
  	return __unsigned_sub_bit(b, a) | b_sign;
 	}
 }
 return 0;
}



static inline fp8x23 fp32_sub(fp8x23 a, fp8x23 b) {
 fp8x23 a_sign = a & 0x80000000;
 fp8x23 b_sign = b & 0x80000000;
  
  a &= 0x7FFFFFFF;
  b &= 0x7FFFFFFF;
  
 //inf nan
 if(a >= 0x7F800000 || b >= 0x7F800000) {
  if(a == 0x7F800000 || b == 0x7F800000)
   return 0x7F800000;
  else
   return 0x7F800001;
 }
 
 if(a_sign == b_sign) {
  if(a > b)
  	return __unsigned_sub_bit(a, b) | a_sign;
  else if(b > a)
  	return __unsigned_sub_bit(b, a) | (!a_sign ? 0x80000000 : 0);
  else
   return 0;
 } else {
 	 return __unsigned_add_bit(a, b) | a_sign;
 }
 return 0;
}



static inline fp8x23 fp32_mul(fp8x23 a, fp8x23 b) {
	fp8x23 a_bits, b_bits, out_bits, sign, inexact;
	int32_t exponent;
	uint64_t mantissa;
	
 a_bits = a;
	b_bits = b;
	
 //sign bit
 // +, + = +
 // -, - = +
 // +, - = -
 // -, + = -
	sign = ((a_bits & 0x80000000) ^ (b_bits & 0x80000000));

 a_bits &= 0x7FFFFFFF;
	b_bits &= 0x7FFFFFFF;

 if(a_bits == 0 || b_bits == 0)
	 return 0;

 if(b_bits == 0x3f800000)
  return a ^ (b_bits & 0x80000000);
	if(a_bits == 0x3f800000)
 	return b ^ (a_bits & 0x80000000);
 	
 //inf nan
 if(a_bits >= 0x7F800000 || b_bits >= 0x7F800000) {
  if(a_bits == 0x7F800000 || b_bits == 0x7F800000)
   return 0x7F800000;
  else
   return 0x7F800001;
 }
	
	int32_t a_exponent = (int32_t)((a_bits) >> 23) - 127;
	int32_t b_exponent = (int32_t)((b_bits) >> 23) - 127;

 exponent = a_exponent + b_exponent;
 
 fp8x23 a_mantissa = a_bits & 0x007FFFFF;
 fp8x23 b_mantissa = b_bits & 0x007FFFFF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 
 inexact = 0;
 
 //multiply and round the low mantissa
 mantissa = (uint64_t)a_mantissa * (uint64_t)b_mantissa;
 inexact |= (mantissa & 0x00000000007FFFFF) != 0;
	mantissa >>= 23;
	
 //normalize
 while(mantissa >= (1 << 24)) {
 	inexact |= mantissa & 1;
		mantissa >>= 1;
		exponent++;
 }
 
 if(inexact)
  feraiseexcept(FE_INEXACT);
 
 if(exponent > 127) {
 	feraiseexcept(FE_OVERFLOW);
 	return 0x7F800000; //overflow
 } if(exponent < -126) {
  mantissa |= 1 << 23;
  int shift = -126 - exponent;
  if(shift > 23 && !(shift < 0)) {
   //undeflow
   feraiseexcept(FE_UNDERFLOW);
   return sign;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((fp8x23)mantissa & 0x007FFFFF);
  }
 }
 
 out_bits = sign | ((exponent + 127) << 23) | ((fp8x23)mantissa & 0x007FFFFF);
 return out_bits;
}


static inline fp8x23 fp32_div(fp8x23 a, fp8x23 b) {
	fp8x23 a_bits, b_bits, out_bits, sign;
	int32_t exponent;
	uint64_t mantissa;
	
	a_bits = a;
	b_bits = b;

 //sign bit
 // +, + = +
 // -, - = +
 // +, - = -
 // -, + = -
 sign = ((a_bits & 0x80000000) ^ (b_bits & 0x80000000));

	a_bits &= 0x7FFFFFFF;
 b_bits &= 0x7FFFFFFF;
 	
 if(b_bits == 0x3f800000)
 	return a_bits | sign;

	if(b_bits == 0) {
	 feraiseexcept(FE_DIVBYZERO);
	 return 0x7F800000 | sign; //infinity
	}
	if(a_bits == 0)
 	return 0;

 //inf nan
 if(a_bits >= 0x7F800000 || b_bits >= 0x7F800000) {
  if(a_bits == 0x7F800000 || b_bits == 0x7F800000)
   return 0x7F800000;
  else
   return 0x7F800001;
 }
	
	int32_t a_exponent = (int32_t)((a_bits) >> 23) - 127;
	int32_t b_exponent = (int32_t)((b_bits) >> 23) - 127;

 exponent = a_exponent - b_exponent;
 
 fp8x23 a_mantissa = a_bits & 0x007FFFFF;
 fp8x23 b_mantissa = b_bits & 0x007FFFFF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 

 //divide mantissa
 mantissa = (fp8x23)(a_mantissa);
 mantissa <<= 23;
 mantissa /= (fp8x23)b_mantissa;
 	 
 //normalize
 while((mantissa & (1 << 23)) == 0 && mantissa != 0) {
  mantissa <<= 1;
  exponent--;
 }
 
 if(exponent > 127) {
 	feraiseexcept(FE_OVERFLOW);
 	return 0x7F800000; //overflow
 } if(exponent < -126) {
  mantissa |= 1 << 23;
  int shift = -126 - exponent;
  if(shift > 23 && !(shift < 0)) {
   //undeflow
   feraiseexcept(FE_UNDERFLOW);
   return sign;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((fp8x23)mantissa & 0x007FFFFF);
  }
 }
  
	out_bits = sign | ((exponent + 127) << 23) | ((fp8x23)mantissa & 0x007FFFFF);
 return out_bits;
}


/*
 compare operator
*/

static inline fp8x23 fp32_gt(fp8x23 a, fp8x23 b) {
	fp8x23 a_sign = a & 0x80000000;
 fp8x23 b_sign = b & 0x80000000;
	a &= 0x7FFFFFFF;
 b &= 0x7FFFFFFF;
 if(a <= 0x7F800000 && b <= 0x7F800000) {
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
 
 
static inline fp8x23 fp32_lt(fp8x23 a, fp8x23 b) {
 fp8x23 a_sign = a & 0x80000000;
	fp8x23 b_sign = b & 0x80000000;
	a &= 0x7FFFFFFF;
 b &= 0x7FFFFFFF;
 if(a <= 0x7F800000 && b <= 0x7F800000) {
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
 
static inline fp8x23 fp32_gte(fp8x23 a, fp8x23 b) {
 fp8x23 a_sign = a & 0x80000000;
 fp8x23 b_sign = b & 0x80000000;
	a &= 0x7FFFFFFF;
	b &= 0x7FFFFFFF;
	if(a <= 0x7F800000 && b <= 0x7F800000) {
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
 
 
static inline fp8x23 fp32_lte(fp8x23 a, fp8x23 b) {
	fp8x23 a_sign = a & 0x80000000;
 fp8x23 b_sign = b & 0x80000000;
 a &= 0x7FFFFFFF;
	b &= 0x7FFFFFFF;
 if(a <= 0x7F800000 && b <= 0x7F800000) {
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
 
 
static inline fp8x23 fp32_eq(fp8x23 a, fp8x23 b) {
	fp8x23 sign_a = a & 0x80000000;
	fp8x23 sign_b = b & 0x80000000;
	a &= 0x7FFFFFFF;
 b &= 0x7FFFFFFF;
	if(a <= 0x7F800000 && b <= 0x7F800000) {
 	return a == b && (sign_a == sign_b);
	}
 //nan
	return 0;
}


static inline fp8x23 fp32_neq(fp8x23 a, fp8x23 b) {
 return !fp32_eq(a, b);
}


static inline fp8x23 fp32_longtofloat32(int64_t x) {
 fp8x23 sign, input, exponent, mantissa;
 int32_t msb;
 
 if(x == 0)
  return 0;
 sign = ((x < 0) ? 1 : 0) << 31;
 input = (sign != 0) ? -x : x;
 msb = 31;

 while(msb >= 0 && ((input >> msb) & 1) == 0)
  --msb;
 
 exponent = (msb + 127) << 23;
 mantissa = 0;
 
 if(msb > 0) {
  int shift = msb - 23;
  if(shift >= 0)
   mantissa = (input >> shift) & 0x007FFFFF;
   else
  mantissa = (input << -shift) & 0x007FFFFF;
 }
 return sign | exponent | mantissa;
}


static inline int64_t fp32_floattolong(fp8x23 x) {
	fp8x23 x_bits, mantissa, sign;
 long integer_part;
 
 x_bits = x;
 	
 sign = x_bits & 0x80000000;
	x_bits &= 0x7FFFFFFF;
 	
	//inf, nan
	if(x_bits >= 0x7F800000)
  return 0x7FFFFFFF;
 	
 int32_t exponent = (x_bits >> 23) - 127;
 	
 if(exponent < 0) //0.xxxx, just round to zero
 	return 0;
 	 
 mantissa = x_bits & 0x007FFFFF;
 	
 mantissa |= (1 << 23);
 	
 if(exponent >= 23)
  integer_part = mantissa << (exponent - 23);
 else
  integer_part = mantissa >> (23 - exponent);
 return sign ? -integer_part : integer_part;
}



static inline fp8x23 fp32_trunc(fp8x23 x) {
 fp8x23 x_bits, out_bits, sign, mantissa;

 x_bits = x;
	sign = x_bits & 0x80000000;
 x_bits &= 0x7FFFFFFF;
 	
 //inf, nan
 if(x_bits >= 0x7F800000) {
  return x_bits;
	}
 	
	int32_t exponent = (x_bits >> 23) - 127;
 	
 if(exponent < 0)
 	return 0;
 	 
 mantissa = x_bits & 0x007FFFFF;
 	
 if(exponent >= 23)
  return x; //integral
 	
	fp8x23 mask = 0xFFFFFFFF << (23 - exponent);
 mantissa &= mask;
 out_bits = sign | ((exponent + 127) << 23) | mantissa;
 return out_bits;
}


#endif

