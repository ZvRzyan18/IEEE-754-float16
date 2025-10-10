#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

/*
 3 degree polynomial sine
*/
static const fp8x23 c[8] = {
	0x3633BA63,
	0xB94FF8AA,
	0x3C08886F,
	0xBE2AAAAB,
	
	0x3e22f983, //inv tau
	0x40c90fdb, //tau
	0x3f22f983, //inv pio2
	0x3fc90fdb, //pio2
};

void fp16_sincos(fp5x10 x, fp5x10 *_sin, fp5x10 *_cos) {
	fp5x10 sign;
	fp8x23 mx, q, x2, x3, poly;
	
	sign = x & 0x8000;
	x &= 0x7FFF;
	if(x >= 0x7C00) {
	 *_sin = 0x7C01; //nan
	 *_cos = 0x7C01;
	 return;
	}
	
 mx =    __fp32_tofloat32(x);
	mx =    fp32_sub(mx, fp32_mul(fp32_trunc(fp32_mul(c[4], mx)), c[5]));
 q =     (fp8x23)fp32_floattolong(fp32_mul(mx, c[6]));
	mx =    fp32_sub(mx, fp32_mul(c[7], fp32_longtofloat32((int64_t)q) ));
	sign ^= (q == 2 || q == 3) << 31;
	mx =    (q == 1 || q == 3) ? fp32_sub(c[7], mx) : mx;
	x2 =    fp32_mul(mx, mx);
	x3 =    fp32_mul(x2, mx);
 poly =  fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x2), c[1]), x2), c[2]), x2), c[3]);
	mx =    fp32_add(mx, fp32_mul(x3, poly));
	if(_sin != NULL) //null safety
	*_sin = __fp32_tofloat16(mx) | sign;
	
 mx =    __fp32_tofloat32(x);
	mx =    fp32_sub(mx, fp32_mul(fp32_trunc(fp32_mul(c[4], mx)), c[5]));
 q =     (fp8x23)fp32_floattolong(fp32_mul(mx, c[6]));
	mx =    fp32_sub(mx, fp32_mul(c[7], fp32_longtofloat32((int64_t)q) ));
	sign    = (q == 1 || q == 2) << 15;
	mx =    (q == 0 || q == 2) ? fp32_sub(c[7], mx) : mx;
	x2 =    fp32_mul(mx, mx);
	x3 =    fp32_mul(x2, mx);
 poly =  fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], x2), c[1]), x2), c[2]), x2), c[3]);
	mx =    fp32_add(mx, fp32_mul(x3, poly));
	if(_cos != NULL) //null safety
	*_cos = __fp32_tofloat16(mx) | sign;
}

