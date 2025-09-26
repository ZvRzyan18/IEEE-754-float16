#include "fp16/math.h"
#include "fp16/float16.h"

/*
 square root
 
 sqrt(x) = x ^ 1/2
*/
uint16_t fp16_sqrt(uint16_t x) {
 //Quake III fast inverse sqrt
 uint16_t x_bits;

 x_bits = x;
 	
 //inf, nan
	if((x_bits & 0x7FFF) >= 0x7C00)
 	return x;

 uint16_t x_half = fp16_mul(x, 0x3800);
 uint16_t three_half = 0x3E00;
 	
 x_bits = 0x59BA - (x_bits >> 1);
 x_bits = fp16_mul(x_bits, fp16_sub(three_half, fp16_mul(x_half, fp16_mul(x_bits, x_bits))));
 x_bits = fp16_mul(x_bits, fp16_sub(three_half, fp16_mul(x_half, fp16_mul(x_bits, x_bits))));
 return fp16_mul(x, x_bits);
}

