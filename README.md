# go-avx

AVX(Advanced Vector Extensions) binding for golang.

## Golang code example

```go
package main

import (
	"fmt"

	avx "github.com/monochromegane/go-avx"
)

func main() {
	dim := 8
	x := avx.MmMalloc(dim)
	y := avx.MmMalloc(dim)
	z := avx.MmMalloc(dim)
	defer avx.MmFree(x)
	defer avx.MmFree(y)
	defer avx.MmFree(z)

	for i := 0; i < dim; i++ {
		x[i] = float32(i)
		y[i] = float32(i + 1)
	}

	avx.Add(dim, x, y, z)

	fmt.Printf("%v\n", z) // [1 3 5 7 9 11 13 15]
}
```

## Features

- Add
- Sub
- Mul
- Dot
- EuclideanDistance 

See also `avx_test.go`.

## License

[MIT](https://github.com/monochromegane/go-avx/blob/master/LICENSE)

## Author

[monochromegane](https://github.com/monochromegane)

