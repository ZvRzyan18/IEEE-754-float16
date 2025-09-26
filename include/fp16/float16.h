#ifndef FLOAT_16_H
#define FLOAT_16_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FP16_INFINITY     0x7C00
#define FP16_NEG_INFINITY 0xFC00
#define FP16_NAN          0x7C01

uint16_t fp16_tofloat16(float x);
float fp16_tofloat32(uint16_t x);
uint16_t fp16_longtofloat16(int64_t x);

/*
 compare operator
*/
uint32_t fp16_gt(uint16_t a, uint16_t b);
uint32_t fp16_lt(uint16_t a, uint16_t b);
uint32_t fp16_gte(uint16_t a, uint16_t b);
uint32_t fp16_lte(uint16_t a, uint16_t b);
uint32_t fp16_eq(uint16_t a, uint16_t b);
uint32_t fp16_neq(uint16_t a, uint16_t b);

/*
 arithmetic operator
*/
uint16_t fp16_add(uint16_t a, uint16_t b);
uint16_t fp16_sub(uint16_t a, uint16_t b);
uint16_t fp16_mul(uint16_t a, uint16_t b);
uint16_t fp16_div(uint16_t a, uint16_t b);

#ifdef __cplusplus
}
#endif

#endif


