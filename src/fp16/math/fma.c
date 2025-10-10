#include "fp16/math.h"
#include "fp16/float16.h"

#include <fenv.h>
/*
 fused multiply add
*/
#define RETURN_ZERO 1
#define RETURN_SAME 2
#define RETURN_INF 3
#define RETURN_NAN 4
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
static inline int multiply(fp5x10 a, fp5x10 b, fp5x10 *out_sign, fp5x10 *out_exponent, fp8x23 *out_mantissa, fp5x10 *product) {
	fp5x10 a_bits, b_bits, sign, inexact;
	int16_t exponent;
	fp8x23 mantissa;
	
 a_bits = a;
	b_bits = b;
	
	sign = ((a_bits & 0x8000) ^ (b_bits & 0x8000));

 a_bits &= 0x7FFF;
	b_bits &= 0x7FFF;

	if(a_bits == 0 || b_bits == 0) {
	 *out_sign = 0;
	 *out_exponent = 0;
	 *out_mantissa = 0;
	 return RETURN_ZERO;
	}

 if(b_bits == 0x3C00) {
  *out_sign = (0x8000 & a) ^ (0x8000 & b);
  return RETURN_SAME;
 }
	if(a_bits == 0x3C00) {
	 *out_sign = (0x8000 & a) ^ (0x8000 & b);
 	return RETURN_SAME;
	}
 	
 //inf, nan
 if(a >= 0x7C00 || b >= 0x7C00) {
  if(a == 0x7C00 || b == 0x7C00)
   return RETURN_INF;
  else
   return RETURN_NAN;
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
 
 //dont scale it back to maintain better precision
 mantissa = (fp8x23)a_mantissa * (fp8x23)b_mantissa;
 
 //calculated value
 *out_sign = (0x8000 & a) ^ (0x8000 & b);
 *out_exponent = exponent+15;
 *out_mantissa = mantissa;
 
 inexact = 0;

 while((*out_mantissa) >= (1 << 21)) {
 	inexact |= (*out_mantissa) & 1;
		(*out_mantissa) >>= 1;
		(*out_exponent)++;
 }
 
 if(inexact)
  feraiseexcept(FE_INEXACT);
 
 if(exponent < -14) {
  (*out_mantissa) |= 1 << 20;
  int shift = -14 - exponent;
  if (shift > 10 && !(shift < 0)) {
  //undeflow
  feraiseexcept(FE_UNDERFLOW);
  return RETURN_ZERO;
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
 *product = sign | ((exponent+15) << 10) | ((fp5x10)mantissa & 0x03FF);

 if(exponent > 15) {//overflow
  feraiseexcept(FE_OVERFLOW);
  return RETURN_INF;
 }
 if(exponent < -14) {

  mantissa |= 1 << 10;
  int shift = -14 - exponent;
  if (shift > 10 && !(shift < 0)) {
   //undeflow
   feraiseexcept(FE_UNDERFLOW);;
   return RETURN_ZERO;
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
static inline fp5x10 unsigned_add_bit(fp5x10 exponent, fp8x23 mantissa, fp5x10 b, fp5x10 *grs) {
 fp5x10 b_bits, out_bits, final_exponent, final_mantissa, shift, inexact, grs_count;
 b_bits = b;
	
	int16_t a_exponent = ((int16_t)exponent) - 15;
 int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;
 
 fp8x23 a_mantissa = mantissa;
 fp8x23 b_mantissa = (b_bits & 0x03FF);

 // add leading ones if not subnormal
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((b_mantissa >> (shift-3)) & 0x00000007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else if (a_exponent < b_exponent) {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((a_mantissa >> (shift-3)) & 0x00000007);
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 } else {
 	(*grs) = 0;
 	final_exponent = a_exponent;
 }
 
 //scale up before addition
 final_mantissa = a_mantissa + (b_mantissa << 10);
 inexact |= (final_mantissa & 0x000003FF) != 0;
 final_mantissa = (a_mantissa + (b_mantissa << 10)) >> 10;
 
 grs_count = 0;
 //normalize
 while(final_mantissa >= (1 << 11)) {
  if(grs_count == 0)
   (*grs) = 0;
  if(grs_count < 3)
   (*grs) |= (final_mantissa & 1) << (2 - grs_count);
  inexact |= final_mantissa & 1;
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

/*
 no sign subtraction
*/
static inline fp5x10 unsigned_sub_bit_a(fp5x10 exponent, fp8x23 mantissa, fp5x10 b, fp5x10 *grs) {
 fp5x10 b_bits, out_bits, shift, inexact;
 int16_t final_mantissa, final_exponent;
	b_bits = b;

	int16_t a_exponent = ((int16_t)(exponent)) - 15;
	int16_t b_exponent = (int16_t)(b_bits >> 10) - 15;

 fp8x23 a_mantissa = mantissa;
 fp8x23 b_mantissa = (b_bits & 0x03FF);

  // add leading ones if not subnormal
 if(b_exponent >= -14)
 b_mantissa |= 1 << 10;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((b_mantissa >> (shift-3)) & 0x00000007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((a_mantissa >> (shift-3)) & 0x00000007);
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 }
 
 //scale up before subtraction
 final_mantissa = a_mantissa - (b_mantissa << 10);
 inexact |= (final_mantissa & 0x000003FF) != 0;
 final_mantissa = (a_mantissa - (b_mantissa << 10)) >> 10;
 
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

/*
 no sign subtraction
*/
static inline fp5x10 unsigned_sub_bit_b(fp5x10 a, fp5x10 exponent, fp8x23 mantissa, fp5x10 *grs) {
 fp5x10 a_bits, out_bits, shift, inexact;
 int16_t final_mantissa, final_exponent;
	a_bits = a;

	int16_t a_exponent = (int16_t)(a_bits >> 10) - 15;
	int16_t b_exponent = ((int16_t)(exponent)) - 15;

 fp8x23 a_mantissa = (a_bits & 0x03FF);
 fp8x23 b_mantissa = mantissa;

  // add leading ones if not subnormal
 if(a_exponent >= -14)
 a_mantissa |= 1 << 10;
 
 inexact = 0;
 //shift to align mantissa
 if(a_exponent > b_exponent) {
  shift = (a_exponent - b_exponent);
  inexact |= (b_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((b_mantissa >> (shift-3)) & 0x00000007);
  b_mantissa >>= shift;
  final_exponent = a_exponent;
 } else {
  shift = (b_exponent - a_exponent);
  inexact |= (a_mantissa & ((1 << (shift+1)) - 1)) != 0;
  (*grs) = (fp5x10)((a_mantissa >> (shift-3)) & 0x00000007);
  a_mantissa >>= shift;
  final_exponent = b_exponent;
 }
 
 //scale up before subtraction
	final_mantissa = (a_mantissa << 10) - b_mantissa;
 inexact |= (final_mantissa & 0x000003FF) != 0;
 final_mantissa = ((a_mantissa << 10) - b_mantissa) >> 10;
 
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


/*
 addition with full precision and sign correction
*/
static inline fp5x10 add(fp5x10 sign, fp5x10 exponent, fp8x23 mantissa, fp5x10 product, fp5x10 b, fp5x10 *grs) {
 fp5x10 a_sign = sign;
 fp5x10 b_sign = b & 0x8000;

 b &= 0x7FFF;
 product &= 0x7FFF;
 //inf, nan
 if(b >= 0x7C00) {
  if(b == 0x7C00)
   return 0x7C00;
  else
   return 0x7C01;
 }
  
 if(a_sign == b_sign) {
  return unsigned_add_bit(exponent, mantissa, b, grs) | a_sign;
 } else {
 	if(product > b)	{
   return unsigned_sub_bit_a(exponent, mantissa, b, grs) | a_sign;
  } else	{
  	return unsigned_sub_bit_b(b, exponent, mantissa, grs) | b_sign;
 	}
 }
 return 0;
}



fp5x10 fp16_fma(fp5x10 x, fp5x10 y, fp5x10 z) {

 fp5x10 sign, exponent, product, grs, sum;
 fp8x23 mantissa;
 
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
 	 grs = 0;
 	 sum = fp16_add((x & 0x7FFF) | sign, z);
 	break;
 	case RETURN_INF:
 	 return 0x7C00;
 	break;
 	case RETURN_NAN:
 	 return 0x7C01;
 	break;
 	default:
 	 sum = add(sign, exponent, mantissa, product, z, &grs);
  break;
 }
 
 //https://stackoverflow.com/questions/79108779/why-do-we-need-both-a-round-bit-and-a-sticky-bit-in-ieee-754-floating-point-impl
 fp5x10 guard = (grs >> 2) & 1;
 fp5x10 round = (grs >> 1) & 1;
 fp5x10 sticky = grs & 1;

 fp5x10 out_mantissa = (sum & 0x03FF);
 
 sign = sum & 0x8000;
 exponent = (sum & 0x7FFF) >> 10;
 
 //only add leading one for non subnormal
 if(exponent > 0)
  out_mantissa |= (1 << 10);
 
 fp5x10 increment = 0;

 //inexact rounding
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
   
   //exponent is too large, overflow exponent
   if(exponent > 30) {
    feraiseexcept(FE_OVERFLOW);
    return 0x7C00;
   }
   
  }
 }
 return sign | (exponent << 10) | (out_mantissa & 0x03FF);
}

