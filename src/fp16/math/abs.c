#include "fp16/math.h"
#include "fp16/float16.h"

/*
 absolute value
*/
uint16_t fp16_abs(uint16_t x) {
 uint16_t mx;
 mx = x & 0x7FFF; //set sign bit to zero
 if(mx > 0x7C00) //inf, nan
  return 0x7C01;
 return mx;
}

