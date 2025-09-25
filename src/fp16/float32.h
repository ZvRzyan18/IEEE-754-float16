#ifndef FP16_FLOAT32_H
#define FP16_FLOAT32_H

#include <stdint.h>

/*
 IEEE 754 float 32 
 this is only here to extend the precision of half float
*/
//bias : 127
//significand bit : 23
//max + exp : 127
//min - exp : -126


static inline uint32_t __fp32_tofloat32(uint16_t x) {
 uint32_t sign = x >> 15;
 uint32_t exp16 = (x & 0x7FFF) >> 10;
 uint32_t mant16 = x & 0x03FF;

 uint32_t sign32 = sign << 31;
 uint32_t exp32, mant32;

 if(exp16 == 0) {
  if(mant16 == 0) {
   return 0;
  } else {
   // subnormal
   int shift = 0;
   uint32_t mant = mant16;
   while((mant & 0x400) == 0) {
    mant <<= 1;
    shift++;
   }
   mant &= 0x3FF;
   exp32 = 127 - 15 - shift;
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
 return sign32 | (exp32 << 23) | mant32;
}


static inline uint16_t __fp32_tofloat16(uint32_t x) {
 uint32_t bits = x;
 uint32_t sign = bits >> 31;
 int32_t exp32 = (bits & 0x7FFFFFFF) >> 23;
 uint32_t mant32 = bits & 0x007FFFFF;

 uint16_t sign16 = sign << 15;
 uint16_t exp16 = 0;
 uint16_t mant16 = 0;

 if(exp32 == 0xFF) {
  //inf, nan
  exp16 = 0x1F;
  mant16 = mant32 ? 1 : 0;
 } else if(exp32 > 142) {
  //overflow
  return 0x7C00;
 } else if(exp32 < 113) {
  //subnormal
  if(exp32 < 103) {
   // too small for subnormal
   return 0;
  } else {
   uint32_t mant = mant32 | 0x800000;
   int shift = 113 - exp32;
   mant16 = mant >> (shift + 13);
   // round to nearest
   if((mant >> (shift + 12)) & 1) {
    mant16 += 1;
   }
  }
 } else {
  exp16 = exp32 - 127 + 15;
  mant16 = mant32 >> 13;
  // round to nearest
  if((mant32 >> 12) & 1) {
   mant16 += 1;
   if(mant16 == 0x400) {
    // rounding overflow in mantissa
    mant16 = 0;
    exp16 += 1;
    if(exp16 >= 0x1F) {
     // overflow after rounding
     exp16 = 0x1F;
     mant16 = 0;
    }
   }
  }
 }
 return sign16 | (exp16 << 10) | mant16;
}



static inline uint32_t __unsigned_add_bit(uint32_t a, uint32_t b) {
 uint32_t a_bits, b_bits, out_bits, final_exponent, final_mantissa;
 a_bits = a;
 b_bits = b;
	
	int32_t a_exponent = (int32_t)(a_bits >> 23) - 127;
 int32_t b_exponent = (int32_t)(b_bits >> 23) - 127;
 
 uint32_t a_mantissa = (a_bits & 0x007FFFFF);
 uint32_t b_mantissa = (b_bits & 0x007FFFFF);

 // add leading ones
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  b_mantissa >>= (a_exponent - b_exponent);
  final_exponent = a_exponent;
 } else if (a_exponent < b_exponent) {
  a_mantissa >>= (b_exponent - a_exponent);
  final_exponent = b_exponent;
 } else {
 	final_exponent = a_exponent;
 }
 
 final_mantissa = a_mantissa + b_mantissa;
 
 //normalize
 while(final_mantissa >= (1 << 24)) {
	 final_mantissa >>= 1;
		final_exponent++;
 }

 out_bits = 0;
 out_bits |= ((final_exponent + 127) << 23) | (final_mantissa & 0x007FFFFF);
 return out_bits;
}



static inline uint32_t __unsigned_sub_bit(uint32_t a, uint32_t b) {
 uint32_t a_bits, b_bits, out_bits;
 int32_t final_mantissa, final_exponent;
 a_bits = a;
	b_bits = b;
	
	if(a_bits == b_bits)	{
	 out_bits = 0;
 	return out_bits;
 }

	int32_t a_exponent = (int32_t)(a_bits >> 23) - 127;
	int32_t b_exponent = (int32_t)(b_bits >> 23) - 127;

 uint32_t a_mantissa = (a_bits & 0x007FFFFF);
 uint32_t b_mantissa = (b_bits & 0x007FFFFF);
 
 // add leading ones
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  b_mantissa >>= (a_exponent - b_exponent);
  final_exponent = a_exponent;
 } else {
  a_mantissa >>= (b_exponent - a_exponent);
  final_exponent = b_exponent;
 }

 final_mantissa = a_mantissa - b_mantissa;
 
 //normalize
 while((final_mantissa & (1 << 23)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 out_bits = ((final_exponent + 127) << 23) | (final_mantissa & 0x007FFFFF);
 return out_bits;
}


static inline uint32_t fp32_add(uint32_t a, uint32_t b) {
 uint32_t a_sign = a & 0x80000000;
 uint32_t b_sign = b & 0x80000000;
  
 a &= 0x7FFFFFFF;
 b &= 0x7FFFFFFF;
 
 //inf nan
 if(a >= 0x7F800000 || b >= 0x7F800000) {
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



static inline uint32_t fp32_sub(uint32_t a, uint32_t b) {
 uint32_t a_sign = a & 0x80000000;
 uint32_t b_sign = b & 0x80000000;
  
 if(a == b)
  return 0;
  
  a &= 0x7FFFFFFF;
  b &= 0x7FFFFFFF;
  
 //inf nan
 if(a >= 0x7F800000 || b >= 0x7F800000) {
  return 0x7F800001;
 }
  
 if(a_sign == b_sign) {
  if(!a_sign && !b_sign) { //both possitive
  	if(a > b) 
   	return __unsigned_sub_bit(a, b);
   else
    return __unsigned_sub_bit(b, a) | 0x80000000;
  } else { //both negative
  	if(a > b) 
   	return __unsigned_sub_bit(a, b) | a_sign;
   else
    return __unsigned_sub_bit(b, a);
   }
  } else {
  	if(a > b) {
  	 if(b_sign)
  	  return __unsigned_add_bit(a, b);
   return __unsigned_add_bit(a, b) | 0x80000000;
  } else {
  	if(a_sign)
  	 return __unsigned_add_bit(a, b) | 0x80000000;
  	return __unsigned_add_bit(a, b);
  }
 }
 return 0;
}



static inline uint32_t fp32_mul(uint32_t a, uint32_t b) {
	uint32_t a_bits, b_bits, out_bits, sign;
	int32_t exponent;
	uint64_t mantissa;
	
 a_bits = a;
	b_bits = b;
	
	if(a_bits == 0 || b_bits == 0)
	 return 0;

 //sign bit
 // +, + = +
 // -, - = +
 // +, - = -
 // -, + = -
	sign = ((a_bits & 0x80000000) ^ (b_bits & 0x80000000));

 a_bits &= 0x7FFFFFFF;
	b_bits &= 0x7FFFFFFF;
 
 if(b_bits == 0x3f800000)
  return a ^ (b_bits & 0x80000000);
	if(a_bits == 0x3f800000)
 	return b ^ (a_bits & 0x80000000);
 	
 //inf nan
 if(a_bits >= 0x7F800000 || b_bits >= 0x7F800000) {
 	return 0x7F800001;
 }
	
	int32_t a_exponent = (int32_t)((a_bits) >> 23) - 127;
	int32_t b_exponent = (int32_t)((b_bits) >> 23) - 127;

 exponent = a_exponent + b_exponent;
 
 uint32_t a_mantissa = a_bits & 0x007FFFFF;
 uint32_t b_mantissa = b_bits & 0x007FFFFF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 

 //multiply and round the low mantissa
 mantissa = (uint64_t)a_mantissa * (uint64_t)b_mantissa;
	mantissa >>= 23;
	
 //normalize
 while(mantissa >= (1 << 24)) {
		mantissa >>= 1;
		exponent++;
 }
 
 if(exponent > 127) {
 	return 0x7F800000; //overflow
 } if(exponent < -126) {
  mantissa |= 1 << 23;
  int shift = -126 - exponent;
  if(shift > 23 && !(shift < 0)) {
   //undeflow
   return 0x7F800000 | 0x80000000;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((uint32_t)mantissa & 0x007FFFFF);
  }
 }
 
 out_bits = sign | ((exponent + 127) << 23) | ((uint32_t)mantissa & 0x007FFFFF);
 return out_bits;
}


static inline uint32_t fp32_div(uint32_t a, uint32_t b) {
	uint32_t a_bits, b_bits, out_bits, sign;
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
 	return a ^ (b_bits & 0x80000000);

	if(b_bits == 0)
	 return 0x7F800000 | sign; //infinity
	if(a_bits == 0)
 	return 0;

 //inf nan
 if(a_bits >= 0x7F800000 || b_bits >= 0x7F800000) {
 	return 0x7F800001;
 }
	
	int32_t a_exponent = (int32_t)((a_bits) >> 23) - 127;
	int32_t b_exponent = (int32_t)((b_bits) >> 23) - 127;

 exponent = a_exponent - b_exponent;
 
 uint32_t a_mantissa = a_bits & 0x007FFFFF;
 uint32_t b_mantissa = b_bits & 0x007FFFFF;
 
 //add leading one to mantissa 1.(mantissa value)
 if(a_exponent >= -126)
 a_mantissa |= 1 << 23;
 if(b_exponent >= -126)
 b_mantissa |= 1 << 23;
 

 //divide mantissa
 mantissa = (uint32_t)(a_mantissa);
 mantissa <<= 23;
 mantissa /= (uint32_t)b_mantissa;
 	 
 //normalize
 while((mantissa & (1 << 23)) == 0 && mantissa != 0) {
  mantissa <<= 1;
  exponent--;
 }
 
 if(exponent > 127) {
 	return 0x7F800000; //overflow
 } if(exponent < -126) {
  mantissa |= 1 << 23;
  int shift = -126 - exponent;
  if(shift > 23 && !(shift < 0)) {
   //undeflow
   return 0x7F800000 | 0x80000000;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((uint32_t)mantissa & 0x007FFFFF);
  }
 }
  
	out_bits = sign | ((exponent + 127) << 23) | ((uint32_t)mantissa & 0x007FFFFF);
 return out_bits;
}


/*
 compare operator
*/

static inline uint32_t fp32_gt(uint32_t a, uint32_t b) {
	uint32_t a_sign = a & 0x80000000;
 uint32_t b_sign = b & 0x80000000;
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
 
 
static inline uint32_t fp32_lt(uint32_t a, uint32_t b) {
 uint32_t a_sign = a & 0x80000000;
	uint32_t b_sign = b & 0x80000000;
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
 
static inline uint32_t fp32_gte(uint32_t a, uint32_t b) {
 uint32_t a_sign = a & 0x80000000;
 uint32_t b_sign = b & 0x80000000;
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
 
 
static inline uint32_t fp32_lte(uint32_t a, uint32_t b) {
	uint32_t a_sign = a & 0x80000000;
 uint32_t b_sign = b & 0x80000000;
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
 
 
static inline uint32_t fp32_eq(uint32_t a, uint32_t b) {
	uint32_t sign_a = a & 0x80000000;
	uint32_t sign_b = b & 0x80000000;
	a &= 0x7FFFFFFF;
 b &= 0x7FFFFFFF;
	if(a <= 0x7F800000 && b <= 0x7F800000) {
 	return a == b && (sign_a == sign_b);
	}
 //nan
	return 0;
}


static inline uint32_t fp32_neq(uint32_t a, uint32_t b) {
 return !fp32_eq(a, b);
}


static inline uint32_t fp32_longtofloat32(int64_t x) {
 uint32_t sign, input, exponent, mantissa;
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


static inline int64_t fp32_floattolong(uint32_t x) {
	uint32_t x_bits, mantissa, sign;
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



static inline uint32_t fp32_trunc(uint32_t x) {
 uint32_t x_bits, out_bits, sign, mantissa;

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
 	
	uint32_t mask = 0xFFFFFFFF << (23 - exponent);
 mantissa &= mask;
 out_bits = sign | ((exponent + 127) << 23) | mantissa;
 return out_bits;
}


#endif

