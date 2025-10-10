#ifndef FLOAT_16_H
#define FLOAT_16_H

#include "fp16/fptypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FP16_INFINITY     0x7C00
#define FP16_NEG_INFINITY 0xFC00
#define FP16_NAN          0x7C01

fp5x10 fp16_tofloat16(float x);
float fp16_tofloat32(fp5x10 x);
fp5x10 fp16_longtofloat16(int64_t x);

uint32_t fp16_isinf(fp5x10 x);
uint32_t fp16_isnan(fp5x10 x);
uint32_t fp16_isnormal(fp5x10 x);
uint32_t fp16_issubnormal(fp5x10 x);
/*
 compare operator
*/
uint32_t fp16_gt(fp5x10 a, fp5x10 b);
uint32_t fp16_lt(fp5x10 a, fp5x10 b);
uint32_t fp16_gte(fp5x10 a, fp5x10 b);
uint32_t fp16_lte(fp5x10 a, fp5x10 b);
uint32_t fp16_eq(fp5x10 a, fp5x10 b);
uint32_t fp16_neq(fp5x10 a, fp5x10 b);

/*
 arithmetic operator
*/
fp5x10 fp16_add(fp5x10 a, fp5x10 b);
fp5x10 fp16_sub(fp5x10 a, fp5x10 b);
fp5x10 fp16_mul(fp5x10 a, fp5x10 b);
fp5x10 fp16_div(fp5x10 a, fp5x10 b);

#ifdef __cplusplus
}
#endif

#endif


