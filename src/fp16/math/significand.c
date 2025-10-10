#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 normalize mantissa as a valid float value
 in a range of [1.0, 2.0]
 
 1.xxx (mantissa value)
*/
fp5x10 fp16_significand(fp5x10 x) {
	if((x & 0x7FFF) > 0x7C00) //inf, nan
	 return 0;
	return (15 << 10) | (x & 0x03FF);
}

