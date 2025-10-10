#ifndef FP16_MATH_H
#define FP16_MATH_H

#include "fp16/fptypes.h"

#ifdef __cplusplus
extern "C" {
#endif
/*
 math functions
*/

fp5x10 fp16_abs(fp5x10 x);
int64_t  fp16_lrint(fp5x10 x);
fp5x10 fp16_rint(fp5x10 x);
fp5x10 fp16_trunc(fp5x10 x);
fp5x10 fp16_floor(fp5x10 x);
fp5x10 fp16_ceil(fp5x10 x);
fp5x10 fp16_round(fp5x10 x);

fp5x10 fp16_sin(fp5x10 x);
fp5x10 fp16_cos(fp5x10 x);
fp5x10 fp16_tan(fp5x10 x);
fp5x10 fp16_asin(fp5x10 x);
fp5x10 fp16_acos(fp5x10 x);
fp5x10 fp16_atan(fp5x10 x);
fp5x10 fp16_atan2(fp5x10 y, fp5x10 x);
void     fp16_sincos(fp5x10 x, fp5x10 *_sin, fp5x10 *_cos);

fp5x10 fp16_hypot(fp5x10 x, fp5x10 y);

fp5x10 fp16_exp2(fp5x10 x);
fp5x10 fp16_exp10(fp5x10 x);
fp5x10 fp16_exp(fp5x10 x);

fp5x10 fp16_log2(fp5x10 x);
fp5x10 fp16_log10(fp5x10 x);
fp5x10 fp16_log(fp5x10 x);

fp5x10 fp16_sqrt(fp5x10 x);
fp5x10 fp16_cbrt(fp5x10 x);

fp5x10 fp16_pow(fp5x10 x, fp5x10 y);

fp5x10 fp16_scalb(fp5x10 x, fp5x10 y);
fp5x10 fp16_scalbn(fp5x10 x, int y);
fp5x10 fp16_significand(fp5x10 x);
int      fp16_ilogb(fp5x10 x);
fp5x10 fp16_fma(fp5x10 x, fp5x10 y, fp5x10 z);

/*

//unimplemented
fp5x10 fp16_fmod(fp5x10 x, fp5x10 y);
fp5x10 fp16_remainder(fp5x10 x, fp5x10 y);
fp5x10 fp16_remquo(fp5x10 x, fp5x10 y, int *r);


fp5x10 fp16_sinh(fp5x10 x);
fp5x10 fp16_cosh(fp5x10 x);
fp5x10 fp16_tanh(fp5x10 x);
fp5x10 fp16_asinh(fp5x10 x);
fp5x10 fp16_acosh(fp5x10 x);
fp5x10 fp16_atanh(fp5x10 x);


fp5x10 fp16_tgamma(fp5x10 x);
fp5x10 fp16_lgamma(fp5x10 x);

fp5x10 fp16_frexp(fp5x10 x, int *exponent);
fp5x10 fp16_ldexp(fp5x10 x, int exponent);

fp5x10 fp16_fdim(fp5x10 x, fp5x10 y);
fp5x10 fp16_copysign(fp5x10 x, fp5x10 y);
fp5x10 fp16_drem(fp5x10 x, fp5x10 y);

*/
#ifdef __cplusplus
}
#endif

#endif


