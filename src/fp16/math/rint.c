#include "fp16/math.h"
#include "fp16/float16.h"

#include <fenv.h>

/*
 round based on the current rounding mode
 rint()/lrint() functions
*/


uint16_t fp16_rint(uint16_t x) {
 switch(fegetround()) {
 	case FE_TOWARDZERO:
 	 return fp16_trunc(x);
 	break;
 	case FE_DOWNWARD:
 	 return fp16_floor(x);
 	break;
 	case FE_UPWARD:
 	 return fp16_ceil(x);
 	break;
 	case FE_TONEAREST:
 	 return fp16_round(x);
 	break;
 }
 return x;
}

