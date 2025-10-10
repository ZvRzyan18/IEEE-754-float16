#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 atan 4 degree polynomial
*/
static const fp8x23 c[7] = {
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

//pi 0x4248
//base arc tangent
static inline fp8x23 __atan(fp8x23 x) {
	fp8x23 mx, x2, x3, poly, high, sign;
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


fp5x10 fp16_atan2(fp5x10 y, fp5x10 x) {
 
 fp5x10 mx, my, x_sign, y_sign;
 fp8x23 ratio;
 
 mx = x & 0x7FFF;
 my = y & 0x7FFF;
 x_sign = x & 0x8000;
 y_sign = y & 0x8000;
 
 /*
  special cases
 */
 if(mx == 0 && my == 0)
  return 0;
 
 //either x or y is nan or inf
 if(mx >= 0x7C00 || my >= 0x7C00) {
  if(mx > 0x7C00 || my > 0x7C00)
   return 0x7C01;
  if(mx < 0x7C00 && my == 0x7C00 && !y_sign)
   return x_sign;
  if(mx < 0x7C00 && my == 0x7C00 && y_sign)
   return 0x4248 | x_sign;
  if(mx == 0x7C00 && my == 0x7C00 && !y_sign) 
   return 0x3A48 | x_sign; //pio4
  if(mx == 0x7C00 && my == 0x7C00 && y_sign)
   return 0x40B6 | x_sign; //3 pi / 4
  if(mx == 0x7C00 && my > 0 && my < 0x7C00)
   return 0x3E48 | x_sign;
 }
 
 if(mx == 0 && (my > 0 && !y_sign))
  return x_sign;
 if(mx == 0 && (my > 0 && y_sign))
  return 0x4248 | x_sign; //pi, x sign
 if((mx <= 0x7C00 && mx > 0) && my == 0)
  return 0x3E48 | x_sign; //pio2, x sign


 //place it here so it would never raise division by zero
 ratio = fp32_div(__fp32_tofloat32(y), __fp32_tofloat32(x));


 //handle full circle with quarant correction
 if((y & 0x8000) && (x & 0x8000))
  return __fp32_tofloat16(fp32_sub(__atan(ratio), 0x40490fdb));
 else if(!(y & 0x8000) && (x & 0x8000))
  return __fp32_tofloat16(fp32_add(__atan(ratio), 0x40490fdb));
 return __fp32_tofloat16(__atan(ratio));
}
