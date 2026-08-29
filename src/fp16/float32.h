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
 fp8x23 sign = (uint32_t)(x & 0x8000) << 16;
 fp8x23 exponent  = (x >> 10) & 0x1F;
 fp8x23 mantissa = x & 0x03FF;

 if(exponent == 0) {
  if(mantissa == 0) {
   return sign; //preserve sign
  }
  //subnormal
  int shift = 0;
  while((mantissa & 0x0400) == 0) {
   mantissa <<= 1;
   shift++;
  }
  mantissa &= 0x03FF;
  int out_exponent = (uint32_t)(127 - 14 - shift);
  fp8x23 out_mantissa = mantissa << 13;
  return sign | (out_exponent << 23) | out_mantissa;
 }
 if(exponent == 0x1F) {
  fp8x23 out_mantissa = mantissa << 13;
  return sign | (0xFF << 23) | out_mantissa;
 }
 int out_exponent = exponent - 15 + 127;
 fp8x23 out_mantissa = mantissa << 13;
 return sign | (out_exponent << 23) | out_mantissa;
}


static inline fp5x10 __fp32_tofloat16(fp8x23 x) {
 fp5x10 sign = (uint16_t)((x >> 16) & 0x8000);
 fp8x23 exp  = (x >> 23) & 0xFF;
 fp8x23 frac = x & 0x7FFFFF;
 if(exp == 0xFF) {
  if(frac == 0)
   return sign | 0x7C00;
  //non-zero mantissa will result in NAN
  return sign | 0x7C00 | (frac >> 13);
 }
 if(exp == 0) {
  if (frac != 0)
   feraiseexcept(FE_UNDERFLOW | FE_INEXACT);
   return sign;
  }
 if(exp >= 143) {
  //round infinity
  feraiseexcept(FE_OVERFLOW | FE_INEXACT);
  switch (fegetround()) {
   case FE_TOWARDZERO:
    return sign | 0x7BFF;
   case FE_UPWARD:
    return sign ? (sign | 0x7BFF) : 0x7C00;
   case FE_DOWNWARD:
    return sign ? 0xfC00 : 0x7BFF;
   case FE_TONEAREST:
    return sign | 0x7C00;
   default:
    return sign | 0x7C00;
   }
  }
 fp8x23 sig = frac | 0x800000;

 // exponent >= 113 : normal
 // exponent < 113 subnormal
 if(exp >= 113) {
  uint16_t h_exp = (uint16_t)(exp - 112);
  fp8x23 h_frac = sig >> 13;
  fp8x23 inexact = sig & 0x1fff;
  if(inexact)
   feraiseexcept(FE_INEXACT);
  fp8x23 guard  = (inexact >> 12) & 1;
  fp8x23 round  = (inexact >> 11) & 1;
  fp8x23 sticky = inexact & 0x7ff;

  fp8x23 increment = 0;
  switch(fegetround()) {
   case FE_TONEAREST:
    increment = guard && (round || sticky || (h_frac & 1));
   break;
   case FE_TOWARDZERO:
   break;
   case FE_UPWARD:
    increment = !sign && inexact != 0;
   break;
   case FE_DOWNWARD:
    increment = sign && inexact != 0;
   break;
   default:
    increment = guard && (round || sticky || (h_frac & 1));
   break;
  }

  if(increment) {
   h_frac++;
   if(h_frac == 0x400) {
    h_frac = 0;
    h_exp++;
    if(h_exp == 31) {
     feraiseexcept(FE_OVERFLOW);
     return sign | 0x7C00;
    }
   }
  }
  return sign | (h_exp << 10) | (h_frac & 0x3ff);
 }
 //2^-14 is zero exponent = subnormal
 //since there is 10 mantissa, the subnormal
 //can reach up to 2^-24
 fp8x23 shift = 126 - exp;

 fp8x23 h_frac = 0;
 fp8x23 inexact;

 if(shift < 32) {
  h_frac = sig >> shift;
  inexact = sig & ((1u << shift) - 1);
 } else {
  inexact = sig;
 }

 if(inexact)
  feraiseexcept(FE_INEXACT);
 uint32_t increment = 0;
 if(inexact) {
  fp8x23 halfway = 1u << (shift - 1);
  switch (fegetround()) {
  case FE_TONEAREST:
   increment = (inexact > halfway) || (inexact == halfway && (h_frac & 1));
  break;
  case FE_TOWARDZERO:
  break;
  case FE_UPWARD:
   increment = !sign;
  break;
  case FE_DOWNWARD:
   increment = sign;
  break;
  default:
   increment = (inexact > halfway) || (inexact == halfway && (h_frac & 1));
  break;
  }
 }
 if(increment) {
  h_frac++;
  if(h_frac == 0x400)
   return sign | 0x0400;
  }
 return sign | (fp5x10)h_frac;
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

