# CubeCL Linear Algebra Library.


The crate contains common linear algebra algorithms.

## Algorithms

- [X] Tiling 2D Matrix Multiplication.

  The kernel is very flexible and can be used on pretty much any hardware.
- [X] Cooperative Matrix Multiplication.

  The kernel is using Automatic Mixed Precision (AMP) to leverage cooperative matrix-multiply and accumulate instructions.
  For `f32` tensors, the inputs are casted into `f16`, but the accumulation is still performed in `f32`.
  This may cause a small lost in precision, but with way faster execution.

## Benchmarks

You can run the benchmarks from the workspace with the following:

```bash
cargo bench --bench matmul --features wgpu # for wgpu
cargo bench --bench matmul --features cuda # for cuda
```

On an RTX 3070 we get the following results:

```
matmul-wgpu-f32-tiling2d

―――――――― Result ―――――――――
  Samples     100
  Mean        13.289ms
  Variance    28.000ns
  Median      13.271ms
  Min         12.582ms
  Max         13.768ms
―――――――――――――――――――――――――
matmul-cuda-f32-tiling2d

―――――――― Result ―――――――――
  Samples     100
  Mean        12.754ms
  Variance    93.000ns
  Median      12.647ms
  Min         12.393ms
  Max         14.501ms
―――――――――――――――――――――――――
matmul-cuda-f32-cmma

―――――――― Result ―――――――――
  Samples     100
  Mean        4.996ms
  Variance    35.000ns
  Median      5.084ms
  Min         4.304ms
  Max         5.155ms
―――――――――――――――――――――――――
```

