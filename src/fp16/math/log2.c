#include "fp16/math.h"
#include "fp16/float16.h"
#include "fp16/float32.h"

static const uint32_t c[10] = {
	0xBC14E9A5,
	0x3DFDA2D1,
	0xBF3F945A,
	0x4028E1AC,
	0xC0C06E0A,
	0x41149364,
	0xC1201D3A,
	0x41027A14,
	0xC05B39A4,
	0x3F800000,
};

/*
 logarithm base 2
*/
static inline uint32_t __log2(uint32_t x) {
 int32_t whole;
 uint32_t m, poly;
 
 m = 1065353216U | (x & 0x007FFFFF);
 whole = (x >> 23) - 127;
	poly = fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(fp32_add(fp32_mul(c[0], m), c[1]), m), c[2]), m), c[3]), m), c[4]), m), c[5]), m), c[6]), m), c[7]), m), c[8]);
 return fp32_add(fp32_longtofloat32(whole), poly);
}


uint16_t fp16_log2(uint16_t x) {
	uint16_t sign;
	uint32_t mx;
	
	sign = x & 0x8000;
	if(sign) 
	 return 0x7C01;
	if((x & 0x7FFF) >= 0x7C00) //inf, nan
	 return x;
	mx = __fp32_tofloat32(x);
	if((mx >> 23) < 127) {
	 mx = fp32_div(c[9], mx);
	 sign = 0x8000;
	}
 return __fp32_tofloat16(__log2(mx)) | sign;
}

