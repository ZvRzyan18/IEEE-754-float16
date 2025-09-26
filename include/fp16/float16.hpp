#ifndef FLOAT_16_HPP
#define FLOAT_16_HPP

#include "fp16/float16.h"
#include "fp16/math.h"

#ifdef __cplusplus

#include <cstdint>
#include <utility>

namespace fp {

/*
 class wrapper
*/
class float16 {
	private:
	 uint16_t bits;
	
	public:
	
	inline float16() {
		this->bits = 0;
	}
	
 explicit	inline float16(float x) {
 	this->bits = fp16_tofloat16(x);
 }
 
 inline ~float16() {
 	this->bits = 0;
 }
 
 explicit inline operator float() {
 	return fp16_tofloat32(this->bits);
 }
 
 inline float16 operator+(float16 a) {
 	float16 out;
 	 out.bits = fp16_add(this->bits, a.bits);
 	return out;
 }
 
 inline float16 operator-(float16 a) {
 	float16 out;
 	 out.bits = fp16_sub(this->bits, a.bits);
 	return out;
 }

 inline float16 operator*(float16 a) {
 	float16 out;
 	 out.bits = fp16_mul(this->bits, a.bits);
 	return out;
 }

 inline float16 operator/(float16 a) {
 	float16 out;
 	 out.bits = fp16_div(this->bits, a.bits);
 	return out;
 }
 
 inline float16 operator-(void) {
 	float16 out;
 	if(this->bits & 0x8000) {
 		out.bits = this->bits & 0x7FFF;
 	} else {
 		out.bits = this->bits | 0x8000;
 	}
 	return out;
 }

 inline bool operator>(float16 a) {
 	return fp16_gt(this->bits, a.bits);
 }

 inline bool operator<(float16 a) {
 	return fp16_lt(this->bits, a.bits);
 }

 inline bool operator>=(float16 a) {
 	return fp16_gte(this->bits, a.bits);
 }

 inline bool operator<=(float16 a) {
 	return fp16_lte(this->bits, a.bits);
 }

 inline bool operator==(float16 a) {
 	return fp16_eq(this->bits, a.bits);
 }

 inline bool operator!=(float16 a) {
 	return fp16_neq(this->bits, a.bits);
 }
};


/*
 special cases
*/
inline float16 infinity() {
 uint16_t out = FP16_INFINITY;
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 neg_infinity() {
 uint16_t out = FP16_NEG_INFINITY;
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 nan() {
 uint16_t out = FP16_NAN;
 return *reinterpret_cast<float16*>(std::addressof(out));
}

/*
 math functions
*/
inline float16 abs(float16 x) {
 uint16_t out = fp16_abs(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline int64_t lrint(float16 x) {
 return fp16_lrint(*reinterpret_cast<uint16_t*>(std::addressof(x)));
}

inline float16 rint(float16 x) {
 uint16_t out = fp16_rint(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 trunc(float16 x) {
 uint16_t out = fp16_trunc(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 floor(float16 x) {
 uint16_t out = fp16_floor(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 ceil(float16 x) {
 uint16_t out = fp16_ceil(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 round(float16 x) {
 uint16_t out = fp16_round(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 sin(float16 x) {
 uint16_t out = fp16_sin(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 cos(float16 x) {
 uint16_t out = fp16_cos(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 tan(float16 x) {
 uint16_t out = fp16_tan(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 asin(float16 x) {
 uint16_t out = fp16_asin(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 acos(float16 x) {
 uint16_t out = fp16_acos(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 atan(float16 x) {
 uint16_t out = fp16_atan(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 atan2(float16 y, float16 x) {
 uint16_t out = fp16_atan2(*reinterpret_cast<uint16_t*>(std::addressof(y)), *reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline void sincos(float16 x, float16 *s, float16 *c) {
 fp16_sincos(*reinterpret_cast<uint16_t*>(std::addressof(x)), reinterpret_cast<uint16_t*>(std::addressof(s)), reinterpret_cast<uint16_t*>(std::addressof(c)));
}

inline float16 hypot(float16 x, float16 y) {
 uint16_t out = fp16_hypot(*reinterpret_cast<uint16_t*>(std::addressof(x)), *reinterpret_cast<uint16_t*>(std::addressof(y)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 exp2(float16 x) {
 uint16_t out = fp16_exp2(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 exp10(float16 x) {
 uint16_t out = fp16_exp10(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 exp(float16 x) {
 uint16_t out = fp16_exp(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 log2(float16 x) {
 uint16_t out = fp16_log2(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 log10(float16 x) {
 uint16_t out = fp16_log10(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 log(float16 x) {
 uint16_t out = fp16_log(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 sqrt(float16 x) {
 uint16_t out = fp16_sqrt(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 cbrt(float16 x) {
 uint16_t out = fp16_atan(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 pow(float16 x, float16 y) {
 uint16_t out = fp16_pow(*reinterpret_cast<uint16_t*>(std::addressof(x)), *reinterpret_cast<uint16_t*>(std::addressof(y)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 scalb(float16 x, float16 y) {
 uint16_t out = fp16_scalb(*reinterpret_cast<uint16_t*>(std::addressof(x)), *reinterpret_cast<uint16_t*>(std::addressof(y)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 scalbn(float16 x, int y) {
 uint16_t out = fp16_scalbn(*reinterpret_cast<uint16_t*>(std::addressof(x)), y);
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline float16 significand(float16 x) {
 uint16_t out = fp16_significand(*reinterpret_cast<uint16_t*>(std::addressof(x)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}

inline int ilogb(float16 x) {
 return fp16_ilogb(*reinterpret_cast<uint16_t*>(std::addressof(x)));
}

inline float16 fma(float16 x, float16 y, float16 z) {
 uint16_t out = fp16_fma(*reinterpret_cast<uint16_t*>(std::addressof(x)), *reinterpret_cast<uint16_t*>(std::addressof(y)), *reinterpret_cast<uint16_t*>(std::addressof(z)));
 return *reinterpret_cast<float16*>(std::addressof(out));
}


} //fp namespace

#endif

#endif

