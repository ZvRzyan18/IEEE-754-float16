# IEEE-754-float16
complete set of floating point operation, including 

 • Arithmetic (+, -, *, /), (`fp16_add`, `fp16_sub`, `fp16_mul`, `fp16_div`)

 • Compare (>, <, <=, >=, ==, !=), 
   (`fp16_gt`, `fp16_lt`, `fp16_lte`, `fp16_gte`, `fp16_eq`, `fp16_neq`)
 
 • Special case handle (inf, nan, subnormal, overflow, undeflow)

 • Rounding (rint, lrint, trunc, ceil, floor, round)
 
 • Math operations (trigonometry, exponental, logarithmic, roots, etc..)

 • Special functions (`fp16_fma`, `fp16_ilogb`, `fp16_hypot`, `fp16_significand`, `fp16_scalb`, `fp16_scalbn`)

## C++ Example
```cpp
 #include "fp16/float16.hpp"
 #include <iostream>

 int main() {
  fp::float16 a = fp::float16(4.5f);
  fp::float16 b = fp::float16(78.9f);
  fp::float16 c = fp::sin(a + b);

  std::cout << static_cast<float>(c);
  return 0;
 }
```
## C Example
```c
 #include "fp16/float16.h"
 #include "fp16/math.h"
 #include <stdio.h>

 int main() {
  uint16_t a = fp16_tofloat16(33.2f);
  uint16_t b = fp16_tofloat16(12.0f);
  uint16_t c = fp16_div(a, b);

  uint16_t d = fp16_fma(a, b, c);
  float out = fp16_tofloat32(d);
  printf("%.3f", out);
  return 0;
 }
```
 

