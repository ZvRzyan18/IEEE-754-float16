#include "fp16/math.h"
#include "fp16/float16.h"

/*
 absolute value
*/
fp5x10 fp16_abs(fp5x10 x) {
 fp5x10 mx;
 mx = x & 0x7FFF; //set sign bit to zero
 if(mx > 0x7C00) //inf, nan
  return x;
 return mx;
}

