package avx

import (
	"math"
	"sync"
	"testing"
)

func TestMmMalloc(t *testing.T) {
	for _, size := range []int{7, 8, 15} {
		func(size int) {
			x := MmMalloc(size)
			defer MmFree(x)

			if len(x) != align(size) {
				t.Errorf("MmMalloc should return float slice size of %d, but size is %d", align(size), len(x))
			}
		}(size)
	}
}

func TestAdd(t *testing.T) {
	for _, size := range []int{7, 8, 15} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			z := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)
			defer MmFree(z)

			truth := make([]float32, size)
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth[i] = x[i] + y[i]
			}

			Add(size, x, y, z)

			for i := 0; i < size; i++ {
				if truth[i] != z[i] {
					t.Errorf("Add should return %f in %d, but %f", truth[i], i, z[i])
				}
			}
		}(size)
	}
}

func TestSub(t *testing.T) {
	for _, size := range []int{7, 8, 15} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			z := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)
			defer MmFree(z)

			truth := make([]float32, size)
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth[i] = x[i] - y[i]
			}

			Sub(size, x, y, z)

			for i := 0; i < size; i++ {
				if truth[i] != z[i] {
					t.Errorf("Mul should return %f in %d, but %f", truth[i], i, z[i])
				}
			}
		}(size)
	}
}

func TestMul(t *testing.T) {
	for _, size := range []int{7, 8, 15} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			z := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)
			defer MmFree(z)

			truth := make([]float32, size)
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth[i] = x[i] * y[i]
			}

			Mul(size, x, y, z)

			for i := 0; i < size; i++ {
				if truth[i] != z[i] {
					t.Errorf("Mul should return %f in %d, but %f", truth[i], i, z[i])
				}
			}
		}(size)
	}
}

func TestDot(t *testing.T) {
	for _, size := range []int{7, 8, 15} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)

			var truth float32
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth += x[i] * y[i]
			}

			result := Dot(size, x, y)
			if truth != result {
				t.Errorf("Dot should return %f, but %f", truth, result)
			}
		}(size)
	}
}

func TestEuclideanDistance(t *testing.T) {
	for _, size := range []int{7, 8, 15} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)

			var tmp float64
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				tmp += math.Pow(float64(x[i]-y[i]), 2.0)
			}
			truth := float32(math.Sqrt(float64(tmp)))

			result := EuclideanDistance(size, x, y)
			if truth != result {
				t.Errorf("Dot should return %f, but %f", truth, result)
			}
		}(size)
	}
}

func BenchmarkEuclideanDistanceAVX(b *testing.B) {
	size := 2048
	x := MmMalloc(size)
	y := MmMalloc(size)
	defer MmFree(x)
	defer MmFree(y)
	for i := 0; i < size; i++ {
		x[i] = float32(i)
		y[i] = float32(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EuclideanDistance(size, x, y)
	}
}

func BenchmarkEuclideanDistanceGoroutine(b *testing.B) {
	size := 2048
	x := make([]float64, size)
	y := make([]float64, size)
	for i := 0; i < size; i++ {
		x[i] = float64(i)
		y[i] = float64(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		numWorkers := 8
		queue := make(chan int, numWorkers)
		out := make(chan float64, numWorkers)
		go func() {
			for idx, _ := range x {
				queue <- idx
			}
			close(queue)
		}()
		var wg sync.WaitGroup
		for n := 0; n < numWorkers; n++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for q := range queue {
					out <- math.Pow(x[q]-y[q], 2)
				}
			}()
		}
		done := make(chan struct{})
		result := 0.0
		go func() {
			for o := range out {
				result += o
			}
			done <- struct{}{}
		}()
		wg.Wait()
		close(out)
		<-done
		close(done)
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	size := 2048
	x := make([]float64, size)
	y := make([]float64, size)
	for i := 0; i < size; i++ {
		x[i] = float64(i)
		y[i] = float64(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distance := 0.0
		for idx, _ := range x {
			diff := x[idx] - y[idx]
			distance += math.Pow(diff, 2)
		}
		math.Sqrt(distance)
	}
}

func BenchmarkHypot(b *testing.B) {
	size := 2048
	x := make([]float64, size)
	y := make([]float64, size)
	for i := 0; i < size; i++ {
		x[i] = float64(i)
		y[i] = float64(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		norm := 0.0
		for idx, _ := range x {
			diff := x[idx] - y[idx]
			norm = math.Hypot(norm, diff)
		}
	}
}

func TestAlign(t *testing.T) {
	expects := [][]int{
		[]int{7, 8},
		[]int{8, 8},
		[]int{9, 16},
	}
	for _, expect := range expects {
		if size := align(expect[0]); size != expect[1] {
			t.Errorf("align should return %d, but %d", expect[1], size)
		}
	}
}
