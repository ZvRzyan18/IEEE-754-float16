#include "fp16/math.h"
#include "fp16/float16.h"

/*
 fused multiply add
*/
#define RETURN_ZERO 1
#define RETURN_SAME 2
#define RETURN_NAN 3

/*
 # steps for fma(x, y, z)
 
 # multiplication
 • add exponent of x and y
  exponent = ((x >> 10) - 15) + ((y >> 10) - 15)
  
 • add leading one if not subnormal and then
  multiply their mantissa as if they were fixed point.
  
  x_mantissa = x & 0x03FF;
  y_mantissa = x & 0x03FF;
  
  if(!x_subnormal)
   x_mantissa |= (1 << 10);
  if(!y_subnormal)
   y_mantissa |= (1 << 10);
  
  mantissa = (x_mantissa * y_mantissa) >> 10;
  
 • since the significand is 10 bit, it would become 20-22 bit after fixed point multiplication
  to maintain full precision, we should not scale them back
  mantissa = x_mantissa * y_mantissa;
  
 • instead, we are going to normalize them to readjust the value, and the mantissa 
  should remain in 22-bits until we perform addition
  
  -->> normalize.
  
  while(mantissa >= (1 << 21)) {
	 	mantissa >>= 1;
		 exponent++;
  }
  
 • the sign is calculated using EOR operation.
  sign = ((a_bits & 0x8000) ^ (b_bits & 0x8000)) << 15;
  
 • we could assemble them together like so, as an estimation value.
  xy = sign | ((exponent + 15) << 10) | (mantissa >> 10)
  
  
  # for addition xy + z
 
 • add leading one to z's mantissa if it is not subnormal
 
   if(!z_subnormal)
    z_mantissa |= (1 << 10);
   
 • we are going to adjust the exponent of both xy and z
   first, select the maximum between xy and z mantissa
   if we selected xy as max value, we'd shift right the z mantissa by both
   exponent difference and vice versa. if both exponents are equal
   no adjustment needed.
   
  if(xy_exponent > z_exponent) {
   z_mantissa >>= (xy_exponent - z_exponent);
   final_exponent = xy_exponent;
  } else if (xy_exponent < z_exponent) {
   xy_mantissa >>= (b_exponent - xy_exponent);
   final_exponent = z_exponent;
  } else {
  	final_exponent = xy_exponent;
  }
 
 • now we could calculate the final mantissa, and before that, we should align z mantissa first
  z_mantissa <<= 10;
  
 #cases 1
 • and then add if their sign are equal (+, +), (-, -)
  final_mantissa = xy_mantissa + z_mantissa
  
 #cases 2
 • if xy and z sign are not equal (+, -), (-, +), we should subtract them
  final_mantissa = xy_mantissa - z_mantissa

 #cases 3
 • if z mantissa is higher than xy mantissa and their sign are not equal, we should
  swap then and flip the sign
  final_mantissa = z_mantissa - xy_mantissa

 	if(sign) {
 		sign &= (~0x8000);
 	} else {
 		sign |= 0x8000;
 	}
  
 • we could now scale it back to store it into float value
  final_mantissa >>= 10;
  
 • and then normalize
 
 • then return the final float
 
 return sign | ((final_exponent + 15) << 10) | (final_mantissa & 0x03FF);
*/




/*
 bit by bit multiplication without loosing full precision
*/
static inline int multiply(uint16_t a, uint16_t b, uint16_t *out_sign, uint16_t *out_exponent, uint32_t *out_mantissa, uint16_t *product) {
	uint16_t a_bits, b_bits, sign;
	int16_t exponent;
	uint32_t mantissa;
	
 a_bits = a;
	b_bits = b;
	
	if(a_bits == 0 || b_bits == 0) {
	 *out_sign = 0;
	 *out_exponent = 0;
	 *out_mantissa = 0;
	 return RETURN_ZERO;
	}

	sign = ((a_bits & 0x8000) ^ (b_bits & 0x8000));

 a_bits &= 0x7FFF;
	b_bits &= 0x7FFF;
 	
 if(b_bits == 0x3C00) {
  *out_sign = (0x8000 & a) ^ (0x8000 & b);
  return RETURN_SAME;
 }
	if(a_bits == 0x3C00) {
	 *out_sign = (0x8000 & a) ^ (0x8000 & b);
 	return RETURN_SAME;
	}
 	
 //inf nan
 if(a_bits >= 0x7C00 || b_bits >= 0x7C00) {
 	return RETURN_NAN;
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
 
 //dont scale it back to maintain better precision
 mantissa = (uint32_t)a_mantissa * (uint32_t)b_mantissa;
 
 //calculated value
 *out_sign = (0x8000 & a) ^ (0x8000 & b);
 *out_exponent = exponent+15;
 *out_mantissa = mantissa;

 while((*out_mantissa) >= (1 << 21)) {
		(*out_mantissa) >>= 1;
		(*out_exponent)++;
 }
 
 if(exponent < -14) {
  (*out_mantissa) |= 1 << 20;
  int shift = -14 - exponent;
  if (shift > 10 && !(shift < 0)) {
  //undeflow
  return RETURN_NAN;
 } else {
  (*out_mantissa) >>= shift;
  *out_exponent = 0;
 }
}

 //estimated value
	mantissa >>= 10;
 while(mantissa >= (1 << 11)) {
		mantissa >>= 1;
		exponent++;
 }
 *product = sign | ((exponent+15) << 10) | ((uint16_t)mantissa & 0x03FF);

 if(exponent > 15) {//overflow
  return RETURN_NAN;
 }
 if(exponent < -14) {

  mantissa |= 1 << 10;
  int shift = -14 - exponent;
  if (shift > 10 && !(shift < 0)) {
   //undeflow
   return RETURN_NAN;
  } else {
   mantissa >>= shift;
   *product = sign | (mantissa & 0x03FF);
  }
 }

 return 0;
}

/*
 no sign addition
*/
static inline uint16_t unsigned_add_bit(uint16_t exponent, uint32_t mantissa, uint16_t b) {
 uint16_t b_bits, out_bits, final_exponent, final_mantissa;
 b_bits = b;
	
	int16_t a_exponent = ((int16_t)exponent) - 15;
 int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;
 
 uint32_t a_mantissa = mantissa;
 uint32_t b_mantissa = (b_bits & 0x03FF);

 // add leading ones if not subnormal
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
 
 //scale up before addition
 final_mantissa = (a_mantissa + (b_mantissa << 10)) >> 10;

 //normalize
 while(final_mantissa >= (1 << 11)) {
	 final_mantissa >>= 1;
		final_exponent++;
 }
 
 out_bits = 0;
 out_bits |= ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}

/*
 no sign subtraction
*/
static inline uint16_t unsigned_sub_bit_a(uint16_t exponent, uint32_t mantissa, uint16_t b) {
 uint16_t b_bits, out_bits;
 int16_t final_mantissa, final_exponent;
	b_bits = b;

	int16_t a_exponent = ((int16_t)(exponent)) - 15;
	int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;

 uint32_t a_mantissa = mantissa;
 uint32_t b_mantissa = (b_bits & 0x03FF);

  // add leading ones if not subnormal
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
 
 //scale up before subtraction
 final_mantissa = (a_mantissa - (b_mantissa << 10)) >> 10;

 //normalize
 while((final_mantissa & (1 << 10)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 out_bits = ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}

/*
 no sign subtraction
*/
static inline uint16_t unsigned_sub_bit_b(uint16_t a, uint16_t exponent, uint32_t mantissa) {
 uint16_t a_bits, out_bits;
 int16_t final_mantissa, final_exponent;
	a_bits = a;

	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
	int16_t b_exponent = ((int16_t)(exponent)) - 15;

 uint32_t a_mantissa = (a_bits & 0x03FF);
 uint32_t b_mantissa = mantissa;

  // add leading ones if not subnormal
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  b_mantissa >>= (a_exponent - b_exponent);
  final_exponent = a_exponent;
 } else {
  a_mantissa >>= (b_exponent - a_exponent);
  final_exponent = b_exponent;
 }
 
 //scale up before subtraction
	final_mantissa = ((a_mantissa << 10) - b_mantissa) >> 10;
 
 //normalize
 while((final_mantissa & (1 << 10)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 out_bits = ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}


/*
 addition with full precision and sign correction
*/
static inline uint16_t add(uint16_t sign, uint16_t exponent, uint32_t mantissa, uint16_t product, uint16_t b) {
 uint16_t a_sign = sign;
 uint16_t b_sign = b & 0x8000;

 b &= 0x7FFF;
 product &= 0x7FFF;
 //inf nan
 if(b >= 0x7C00) {
  return 0x7C01;
 }
  
 if(a_sign == b_sign) {
  return unsigned_add_bit(exponent, mantissa, b) | a_sign;
 } else {
 	if(product > b)	{
   return unsigned_sub_bit_a(exponent, mantissa, b) | a_sign;
  } else	{
  	return unsigned_sub_bit_b(b, exponent, mantissa) | b_sign;
 	}
 }
 return 0;
}


uint16_t fp16_fma(uint16_t x, uint16_t y, uint16_t z) {

 uint16_t sign, exponent, product;
 uint32_t mantissa;
 
 if(z == 0) {
 	return fp16_mul(x, y);
 }
 
 int error = multiply(x, y, &sign, &exponent, &mantissa, &product);
 
 //handle exact cases
 switch(error) {
 	case RETURN_ZERO:
 	 return z;
 	break;
 	case RETURN_SAME:
 	 return fp16_add((x & 0x7FFF) | sign, z);
 	break;
 	case RETURN_NAN:
 	 return 0x7C01;
 	break;
 }
 
 return add(sign, exponent, mantissa, product, z);
}

