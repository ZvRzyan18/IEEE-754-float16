#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 atan 4 degree polynomial
*/
static const uint32_t c[7] = {
	0xBC9A60D6,
	0x3D8F5DEB,
	0xBE0660F6,
	0x3E4B8783,
	0xBEAAA7D5,
	0x3fc90fdb, //pio2
	0x3F800000, //one
};

/*
 arc tangent 2
          • (x, y)
         /|
        / |
       /  |
      /   |
     /__  |
    /   | |
   •-------
   ^
  angle = atan2(y, x)
 
  y is sine value
  x is cosine value
  
  therefore y/c = tangent
  the value could be reversed by arc tangent
*/

//base arc tangent
static inline uint32_t __atan(uint32_t x) {
	uint32_t mx, x2, x3, poly, high, sign;
	sign = x & 0x80000000;

	x &= 0x7FFFFFFF;

	mx = x;
	
	high = mx > c[6];
	if(high) // |x| > 1.0
	 mx = fp32_div(c[6], mx);
	
	x2 =    fp32_mul(mx, mx);
	x3 =    fp32_mul(x2, mx);
 poly =  fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x2), c[1]), x2), c[2]), x2), c[3]), x2), c[4]);
	mx =    fp32_add(mx, fp32_mul(x3, poly));

 if(high) // |x| > 1.0
  mx = fp32_sub(c[5], mx);
 return mx | sign;
}

// TODO : handle inf, nan correctly
uint16_t fp16_atan2(uint16_t y, uint16_t x) {
 uint32_t ratio;
 ratio = fp32_div(__fp32_tofloat32(y), __fp32_tofloat32(x));
 //handle full circle with quarant correction
 if((y & 0x8000) && (x & 0x8000))
  return __fp32_tofloat16(fp32_sub(__atan(ratio), 0x40490fdb));
 else if(!(y & 0x8000) && (x & 0x8000))
  return __fp32_tofloat16(fp32_add(__atan(ratio), 0x40490fdb));
 return __fp32_tofloat16(__atan(ratio));
}
