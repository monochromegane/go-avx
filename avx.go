package avx

/*
#cgo CFLAGS: -mavx -std=c99
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> //AVX: -mavx
void avx_add(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vz = (__m256 *)z;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    for(size_t i=0; i<end; ++i) {
      vz[i] = _mm256_add_ps(vx[i], vy[i]);
    }
}

void avx_sub(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vz = (__m256 *)z;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    for(size_t i=0; i<end; ++i) {
      vz[i] = _mm256_sub_ps(vx[i], vy[i]);
    }
}

void avx_mul(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vz = (__m256 *)z;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    for(size_t i=0; i<end; ++i) {
      vz[i] = _mm256_mul_ps(vx[i], vy[i]);
    }
}

float avx_dot(const size_t n, float *x, float *y)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    __m256 vsum = {0};
    for(size_t i=0; i<end; ++i) {
      vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx[i], vy[i]));
    }
    __attribute__((aligned(32))) float t[8] = {0};
    _mm256_store_ps(t, vsum);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

float avx_euclidean_distance(const size_t n, float *x, float *y)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    __m256 vsub = {0};
    __m256 vsum = {0};
    for(size_t i=0; i<end; ++i) {
      vsub = _mm256_sub_ps(vx[i], vy[i]);
      vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vsub, vsub));
    }
    __attribute__((aligned(32))) float t[8] = {0};
    _mm256_store_ps(t, vsum);
    return sqrt(t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]);
}
*/
import "C"
import (
	"math"
	"reflect"
	"unsafe"
)

func MmMalloc(size int) []float32 {
	size_ := size
	size = align(size)
	ptr := C._mm_malloc((C.size_t)(C.sizeof_float*size), 32)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(ptr)),
		Len:  size,
		Cap:  size,
	}
	goSlice := *(*[]float32)(unsafe.Pointer(&hdr))
	if size_ != size {
		for i := size_; i < size; i++ {
			goSlice[i] = 0.0
		}
	}
	return goSlice
}

func MmFree(v []float32) {
	C._mm_free(unsafe.Pointer(&v[0]))
}

func Add(size int, x, y, z []float32) {
	size = align(size)
	C.avx_add((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]), (*C.float)(&z[0]))
}

func Mul(size int, x, y, z []float32) {
	size = align(size)
	C.avx_mul((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]), (*C.float)(&z[0]))
}

func Sub(size int, x, y, z []float32) {
	size = align(size)
	C.avx_sub((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]), (*C.float)(&z[0]))
}

func Dot(size int, x, y []float32) float32 {
	size = align(size)
	dot := C.avx_dot((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]))
	return float32(dot)
}

func EuclideanDistance(size int, x, y []float32) float32 {
	size = align(size)
	dot := C.avx_euclidean_distance((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]))
	return float32(dot)
}

func align(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}
