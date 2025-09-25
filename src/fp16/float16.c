#include "fp16/float16.h"
#include "fp16/float32.h"

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
 return __fp32_tofloat16(bits.i);
}



float fp16_tofloat32(uint16_t x) {
 __fp32_bits bits;
 bits.i = __fp32_tofloat32(x);
 return bits.f;
}



uint16_t fp16_longtofloat16(long x) {
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


static inline uint16_t unsigned_add_bit(uint16_t a, uint16_t b) {
 uint16_t a_bits, b_bits, out_bits, final_exponent, final_mantissa;
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
 while(final_mantissa >= (1 << 11)) {
	 final_mantissa >>= 1;
		final_exponent++;
 }

 out_bits = 0;
 out_bits |= ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}



static inline uint16_t unsigned_sub_bit(uint16_t a, uint16_t b) {
 uint16_t a_bits, b_bits, out_bits;
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
 while((final_mantissa & (1 << 10)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 out_bits = ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}



uint16_t fp16_add(uint16_t a, uint16_t b) {
 uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
  
 a &= 0x7FFF;
 b &= 0x7FFF;
  
 //inf nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  return 0x7C01;
 }
  
 if(a_sign == b_sign) {
  return unsigned_add_bit(a, b) | a_sign;
 } else {
 	if(a > b)	{
   return unsigned_sub_bit(a, b) | a_sign;
  } else	{
  	return unsigned_sub_bit(b, a) | b_sign;
 	}
 }
 return 0;
}


uint16_t fp16_sub(uint16_t a, uint16_t b) {
 uint16_t a_sign = a & 0x8000;
 uint16_t b_sign = b & 0x8000;
  
 if(a == b)
  return 0;
  
  a &= 0x7FFF;
  b &= 0x7FFF;
  
 //inf nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  return 0x7C01;
 }
  
 if(a_sign == b_sign) {
  if(!a_sign && !b_sign) { //both possitive
  	if(a > b) 
   	return unsigned_sub_bit(a, b);
   else
    return unsigned_sub_bit(b, a) | 0x8000;
  } else { //both negative
  	if(a > b) 
   	return unsigned_sub_bit(a, b) | a_sign;
   else
    return unsigned_sub_bit(b, a);
   }
  } else {
  	if(a > b) {
  	 if(b_sign)
  	  return unsigned_add_bit(a, b);
   return unsigned_add_bit(a, b) | 0x8000;
  } else {
  	if(a_sign)
  	 return unsigned_add_bit(a, b) | 0x8000;
  	return unsigned_add_bit(a, b);
  }
 }
 return 0;
}


uint16_t fp16_mul(uint16_t a, uint16_t b) {
	uint16_t a_bits, b_bits, out_bits, sign;
	int16_t exponent;
	uint32_t mantissa;
	
 a_bits = a;
	b_bits = b;
	
	if(a_bits == 0 || b_bits == 0)
	 return 0;

 //sign bit
 // +, + = +
 // -, - = +
 // +, - = -
 // -, + = -
	sign = ((a_bits & 0x8000) ^ (b_bits & 0x8000));

 a_bits &= 0x7FFF;
	b_bits &= 0x7FFF;
 	
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
	mantissa >>= 10;
	
 //normalize
 if(mantissa >= (1 << 11)) {
		mantissa >>= 1;
		exponent++;
 }
 
 if(exponent > 15) {
 	return 0x7C00; //overflow
 } if(exponent < -14) {
  mantissa |= 1 << 10;
  int shift = -14 - exponent;
  if(shift > 10 && !(shift < 0)) {
   //undeflow
   return 0xFC00;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((uint16_t)mantissa & 0x03FF);
  }
 }
 
 out_bits = sign | ((exponent + 15) << 10) | ((uint16_t)mantissa & 0x03FF);
 return out_bits;
}
 

uint16_t fp16_div(uint16_t a, uint16_t b) {
	uint16_t a_bits, b_bits, out_bits, sign;
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
 	return a ^ (b_bits & 0x8000);

	if(b_bits == 0)
	 return 0x7C00 | sign; //infinity
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
 	 
 //normalize
 while((mantissa & (1 << 10)) == 0 && mantissa != 0) {
  mantissa <<= 1;
  exponent--;
 }
 
 if(exponent > 15) {
 	return 0x7C00; //overflow
 } if(exponent < -14) {
  mantissa |= 1 << 10;
  int shift = -14 - exponent;
  if(shift > 10 && !(shift < 0)) {
   //undeflow
   return 0xFC00;
  } else {
   //subnormal
   mantissa >>= shift;
   return sign | ((uint16_t)mantissa & 0x03FF);
  }
 }
  
	out_bits = sign | ((exponent + 15) << 10) | ((uint16_t)mantissa & 0x03FF);
 return out_bits;
}



 
