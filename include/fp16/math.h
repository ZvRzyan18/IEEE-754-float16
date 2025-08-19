#ifndef FP16_MATH_H
#define FP16_MATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint16_t fp16_abs(uint16_t x);
long     fp16_lrint(uint16_t x);
uint16_t fp16_rint(uint16_t x);
uint16_t fp16_trunc(uint16_t x);
uint16_t fp16_floor(uint16_t x);
uint16_t fp16_ceil(uint16_t x);
uint16_t fp16_round(uint16_t x);

uint16_t fp16_sin(uint16_t x);
uint16_t fp16_cos(uint16_t x);
uint16_t fp16_tan(uint16_t x);
uint16_t fp16_asin(uint16_t x);
uint16_t fp16_acos(uint16_t x);
uint16_t fp16_atan(uint16_t x);
uint16_t fp16_atan2(uint16_t y, uint16_t x);
void     fp16_sincos(uint16_t x, uint16_t *_sin, uint16_t *_cos);

uint16_t fp16_exp2(uint16_t x);
uint16_t fp16_exp10(uint16_t x);
uint16_t fp16_exp(uint16_t x);

uint16_t fp16_log2(uint16_t x);
uint16_t fp16_log10(uint16_t x);
uint16_t fp16_log(uint16_t x);

uint16_t fp16_sqrt(uint16_t x);
uint16_t fp16_cbrt(uint16_t x);

uint16_t fp16_pow(uint16_t x, uint16_t y);

uint16_t fp16_scalb(uint16_t x, uint16_t y);
uint16_t fp16_scalbn(uint16_t x, int y);
uint16_t fp16_significand(uint16_t x);
int      fp16_ilogb(uint16_t x);
uint16_t fp16_logb(uint16_t x);
uint16_t fp16_fma(uint16_t x, uint16_t y, uint16_t z);

#ifdef __cplusplus
}
#endif

#endif


