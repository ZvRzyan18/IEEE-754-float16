#include "fp16/math.h"
#include "fp16/float16.h"

#define RETURN_ZERO 1
#define RETURN_SAME 2
#define RETURN_NAN 3

/*
 multiplication without loosing full precision
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
 
 mantissa = (uint32_t)a_mantissa * (uint32_t)b_mantissa;
 
 //unrounded value
 *out_sign = (0x8000 & a) ^ (0x8000 & b);
 *out_exponent = exponent+15;
 *out_mantissa = mantissa;

 if((*out_mantissa) >= (1 << 21)) {
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

 //rounded value
	mantissa >>= 10;
 if(mantissa >= (1 << 11)) {
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


static inline uint16_t unsigned_add_bit(uint16_t exponent, uint32_t mantissa, uint16_t b) {
 uint16_t b_bits, out_bits, final_exponent, final_mantissa;
 b_bits = b;
	
	int16_t a_exponent = ((int16_t)exponent) - 15;
 int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;
 
 uint32_t a_mantissa = mantissa;
 uint32_t b_mantissa = (b_bits & 0x03FF);

 // add leading ones
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


static inline uint16_t unsigned_sub_bit_a(uint16_t exponent, uint32_t mantissa, uint16_t b) {
 uint16_t b_bits, out_bits;
 int16_t final_mantissa, final_exponent;
	b_bits = b;

	int16_t a_exponent = ((int16_t)(exponent)) - 15;
	int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;

 uint32_t a_mantissa = mantissa;
 uint32_t b_mantissa = (b_bits & 0x03FF);

 // add leading ones
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
 
 final_mantissa = (a_mantissa - (b_mantissa << 10)) >> 10;

 //normalize
 while((final_mantissa & (1 << 10)) == 0 && final_mantissa != 0) {
  final_mantissa <<= 1;
  final_exponent--;
 }

 out_bits = ((final_exponent + 15) <<  10) | (final_mantissa & 0x03FF);
 return out_bits;
}


static inline uint16_t unsigned_sub_bit_b(uint16_t a, uint16_t exponent, uint32_t mantissa) {
 uint16_t a_bits, out_bits;
 int16_t final_mantissa, final_exponent;
	a_bits = a;

	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
	int16_t b_exponent = ((int16_t)(exponent)) - 15;

 uint32_t a_mantissa = (a_bits & 0x03FF);
 uint32_t b_mantissa = mantissa;

 // add leading ones
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
 addition with full precision
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
 
 //handle cases
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

