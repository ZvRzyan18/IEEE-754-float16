# IEEE-754-float16
complete set of floating point operation, including 

 • arithmetic (+, -, *, /)

 • compare (>, <, <=, >=, ==, !=), 
   (`fp16_gt`, `fp16_lt`, `fp16_lte`, `fp16_gte`, `fp16_eq`, `fp16_neq`)
 
 • special case handle (inf, nan, subnormal, overflow, undeflow)

 • rounding (rint, lrint, trunc, ceil, floor, round)
 
 • math operations (trigonometry, exponental, logarithmic, roots, etc..)

## C++ example
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
 

