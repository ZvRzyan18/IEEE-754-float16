#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 3 degree polynomial sine
*/
static const uint32_t c[8] = {
	0x3633BA63,
	0xB94FF8AA,
	0x3C08886F,
	0xBE2AAAAB,
	
	0x3e22f983, //inv tau
	0x40c90fdb, //tau
	0x3f22f983, //inv pio2
	0x3fc90fdb, //pio2
};

/*
 cosine
          • (x, y) = normalized value
         /|
        / |
       /  |
      /   |
     /__  |
    /   | |
   •-------
   ^
  angle
 
  x = cos(angle)
*/

uint16_t fp16_cos(uint16_t x) {
	uint16_t sign;
	uint32_t mx, q, x2, x3, poly;
	
	//sign handle 
	x &= 0x7FFF;
	if(x >= 0x7C00) //inf, nan
	 return 0x7C01; //nan
	
 mx =    __fp32_tofloat32(x);
 //tau remainder
	mx =    fp32_sub(mx, fp32_mul(fp32_trunc(fp32_mul(c[4], mx)), c[5]));
	//quadrant
 q =     (uint32_t)fp32_floattolong(fp32_mul(mx, c[6]));
 //pio2 range reduction
	mx =    fp32_sub(mx, fp32_mul(c[7], fp32_longtofloat32((int64_t)q) ));
	//correction
	sign    = (q == 1 || q == 2) << 15;
	mx =    (q == 0 || q == 2) ? fp32_sub(c[7], mx) : mx;
	
	//polynomial
	x2 =    fp32_mul(mx, mx);
	x3 =    fp32_mul(x2, mx);
 poly =  fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x2), c[1]), x2), c[2]), x2), c[3]);
	mx =    fp32_add(mx, fp32_mul(x3, poly));
	return __fp32_tofloat16(mx) | sign;
}


