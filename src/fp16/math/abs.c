#include "fp16/math.h"
#include "fp16/float16.h"

uint16_t fp16_abs(uint16_t x) {
 uint16_t mx;
 mx = x & 0x7FFF;
 if(mx > 0x7C00)
  return 0x7C01;
 return mx;
}

