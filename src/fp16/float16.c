#include "fp16/float16.h"
#include "fp16/float32.h"
#include "fp16/fptypes.h"

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
	fp8x23 i;
} __fp32_bits;



fp5x10 fp16_tofloat16(float x) {
 __fp32_bits bits;
 bits.f = x;
 //for implementation of __fp32_tofloat16 see @float32.h 
 return __fp32_tofloat16(bits.i);
}



float fp16_tofloat32(fp5x10 x) {
 __fp32_bits bits;
 //for implementation of __fp32_tofloat32 see @float32.h 
 bits.i = __fp32_tofloat32(x);
 return bits.f;
}



fp5x10 fp16_longtofloat16(int64_t x) {
 fp5x10 sign, input, exponent, mantissa;
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



fp8x23 fp16_isinf(fp5x10 x) {
	return (x & 0x7FFF) == 0x7C00;
}


fp8x23 fp16_isnan(fp5x10 x) {
	return (x & 0x7FFF) > 0x7C00;
}


fp8x23 fp16_isnormal(fp5x10 x) {
	return ((x & 0x7FFF) >> 10) > 0;
}


fp8x23 fp16_issubnormal(fp5x10 x) {
	return (((x & 0x7FFF) >> 10) == 0) && (x & 0x03FF) != 0;
}


/*
 compare operator
*/

fp8x23 fp16_gt(fp5x10 a, fp5x10 b) {
	fp5x10 a_sign = a & 0x8000;
 fp5x10 b_sign = b & 0x8000;
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
 
 
fp8x23 fp16_lt(fp5x10 a, fp5x10 b) {
 fp5x10 a_sign = a & 0x8000;
	fp5x10 b_sign = b & 0x8000;
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
 
fp8x23 fp16_gte(fp5x10 a, fp5x10 b) {
 fp5x10 a_sign = a & 0x8000;
 fp5x10 b_sign = b & 0x8000;
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
 
 
fp8x23 fp16_lte(fp5x10 a, fp5x10 b) {
	fp5x10 a_sign = a & 0x8000;
 fp5x10 b_sign = b & 0x8000;
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
 
 
fp8x23 fp16_eq(fp5x10 a, fp5x10 b) {
	fp5x10 sign_a = a & 0x8000;
	fp5x10 sign_b = b & 0x8000;
	a &= 0x7FFF;
 b &= 0x7FFF;
	if(a <= 0x7C00 && b <= 0x7C00) {
 	return a == b && (sign_a == sign_b);
	}
 //nan
	return 0;
}


fp8x23 fp16_neq(fp5x10 a, fp5x10 b) {
 return !fp16_eq(a, b);
}


/*
 arithmetic operator
*/


static inline fp5x10 unsigned_add_bit(fp5x10 a, fp5x10 b, fp5x10 *grs) {
 fp5x10 a_bits, b_bits, out_bits, final_exponent, final_mantissa, shift, inexact, grs_count;
 a_bits = a;
 b_bits = b;
	
	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
 int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;
 
 fp5x10 a_mantissa = (a_bits & 0x03FF);
 fp5x10 b_mantissa = (b_bits & 0x03FF);

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
  (*grs) = (fp5x10)((b_mantissa >> (shift-3)) & 0x0007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else if (a_exponent < b_exponent) {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((a_mantissa >> (shift-3)) & 0x0007);
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



static inline fp5x10 unsigned_sub_bit(fp5x10 a, fp5x10 b, fp5x10 *grs) {
 fp5x10 a_bits, b_bits, out_bits, shift, inexact;
 int16_t final_mantissa, final_exponent;
 a_bits = a;
	b_bits = b;
	
	if(a_bits == b_bits)	{
	 out_bits = 0;
 	return out_bits;
 }

	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
	int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;

 fp5x10 a_mantissa = (a_bits & 0x03FF);
 fp5x10 b_mantissa = (b_bits & 0x03FF);

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
  (*grs) = (fp5x10)((b_mantissa >> (shift-3)) & 0x0007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((a_mantissa >> (shift-3)) & 0x0007);
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



fp5x10 fp16_add(fp5x10 a, fp5x10 b) {
 fp5x10 a_sign = a & 0x8000;
 fp5x10 b_sign = b & 0x8000;
 fp5x10 grs, sum;
  
 a &= 0x7FFF;
 b &= 0x7FFF;
  
 //inf, nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  if(a == 0x7C00 || b == 0x7C00)
   return 0x7C00;
  else
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
  fp5x10 guard = (grs >> 2) & 1;
  fp5x10 round = (grs >> 1) & 1;
  fp5x10 sticky = grs & 1;

  fp5x10 out_mantissa = sum & 0x03FF;
  fp5x10 exponent = (sum & 0x7FFF) >> 10;
  fp5x10 sign = sum & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
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


fp5x10 fp16_sub(fp5x10 a, fp5x10 b) {
 fp5x10 a_sign = a & 0x8000;
 fp5x10 b_sign = b & 0x8000;
  
 fp5x10 grs, diff;
 
 grs = 0;
 diff = 0;
   
 a &= 0x7FFF;
 b &= 0x7FFF;

 //inf, nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  if(a == 0x7C00 || b == 0x7C00)
   return 0x7C00;
  else
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
  fp5x10 guard = (grs >> 2) & 1;
  fp5x10 round = (grs >> 1) & 1;
  fp5x10 sticky = grs & 1;

  fp5x10 out_mantissa = diff & 0x03FF;
  fp5x10 exponent = (diff & 0x7FFF) >> 10;
  fp5x10 sign = diff & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
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


fp5x10 fp16_mul(fp5x10 a, fp5x10 b) {
	fp5x10 a_bits, b_bits, out_bits, sign, inexact, grs, grs_count;
	int16_t exponent;
	fp8x23 mantissa;
	
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
 	
 //inf, nan
 if(a_bits >= 0x7C00 || b_bits >= 0x7C00) {
  if(a_bits == 0x7C00 || b_bits == 0x7C00)
   return 0x7C00;
  else
   return 0x7C01;
 }
	
	int16_t a_exponent = (int16_t)((a_bits) >> 10) - 15;
	int16_t b_exponent = (int16_t)((b_bits) >> 10) - 15;

 exponent = a_exponent + b_exponent;
 
 fp5x10 a_mantissa = a_bits & 0x03FF;
 fp5x10 b_mantissa = b_bits & 0x03FF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 //multiply and round the low mantissa
 mantissa = (fp8x23)a_mantissa * (fp8x23)b_mantissa;
 grs = (fp5x10)((mantissa >> 7) & 0x00000007);
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

 out_bits = sign | ((exponent + 15) << 10) | ((fp5x10)mantissa & 0x03FF);

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
   out_bits = sign | ((fp5x10)mantissa & 0x03FF);
  }
 }
 
 
 /*
  inexact rounding
 */
  fp5x10 guard = (grs >> 2) & 1;
  fp5x10 round = (grs >> 1) & 1;
  fp5x10 sticky = grs & 1;

  fp5x10 out_mantissa = out_bits & 0x03FF;
  exponent = (out_bits & 0x7FFF) >> 10;
  sign = out_bits & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
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
 

fp5x10 fp16_div(fp5x10 a, fp5x10 b) {
	fp5x10 a_bits, b_bits, out_bits, sign, inexact, grs;
	int16_t exponent;
	fp8x23 mantissa;
	
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

 //inf, nan
 if(a_bits >= 0x7C00 || b_bits >= 0x7C00) {
  if(a_bits == 0x7C00 || b_bits == 0x7C00)
   return 0x7C00;
  else
   return 0x7C01;
 }
	
	int16_t a_exponent = (int16_t)((a_bits) >> 10) - 15;
	int16_t b_exponent = (int16_t)((b_bits) >> 10) - 15;

 exponent = a_exponent - b_exponent;
 
 fp5x10 a_mantissa = a_bits & 0x03FF;
 fp5x10 b_mantissa = b_bits & 0x03FF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 //divide mantissa
 mantissa = (fp8x23)(a_mantissa);
 mantissa <<= 10;
 mantissa /= (fp8x23)b_mantissa;

 //shift to left more so we can catch the lost bits during division
 inexact = (fp5x10)( ((((fp8x23)(a_mantissa) << 20) / (((fp8x23)b_mantissa) << 10)) & 0x000003FF) );
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
   return sign | ((fp5x10)mantissa & 0x03FF);
  }
 }
	out_bits = sign | ((exponent + 15) << 10) | ((fp5x10)mantissa & 0x03FF);

 /*
  inexact rounding
 */
  fp5x10 guard = (grs >> 2) & 1;
  fp5x10 round = (grs >> 1) & 1;
  fp5x10 sticky = grs & 1;

  fp5x10 out_mantissa = out_bits & 0x03FF;
  exponent = (out_bits & 0x7FFF) >> 10;
  sign = out_bits & 0x8000;
  //only add leading one for non subnormal
  if(exponent > 0)
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



 
