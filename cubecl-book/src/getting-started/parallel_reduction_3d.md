# Parallel reduction 3D

The purpose of this example is to demonstrate how to perform a parallel reduction operation on a 3D
tensor using CubeCL. The reduction will sum the elements along the last dimension (depth) of the
tensor, resulting in a 2D tensor.

## A first try

We will start with a simple implementation of a parallel reduction on a 3D tensor. The goal is to
reduce the tensor along the last dimension (depth) by summing the elements. This will result in a 2D
tensor where each element is the sum of the corresponding elements in the depth dimension.

```rust,ignore
{{#include src/bin/v6-gpu.rs:implementation}}
```

This first example intentionally exceeds typical device limits. Inspecting its output buffer reports
an error such as:

```
Failed to read tensor: Several failures at once
Caused by:
  The bytes were never written
Caused by:
  A launch error happened
Caused by:
  Too many resources were requested during launch
Total unit count exceeds maximum.
Requested 16384 units, max units is 1024.
```

Kernel launch failures are attached to output buffers, so the benchmark can print timings without
reporting this error. The error message indicates that the workgroup size exceeds the limits imposed
by the WebGPU backend. The total number of invocations (64 * 256 * 1 = 16384) exceeds the maximum
allowed invocations per workgroup, which is 1024. In other words, the CubeDim size is too large for
the GPU to handle. We need to find another way to parallelize the reduction operation without
exceeding the limits.

## A better approach

To address the issue, we will parallelize with the `CUBE_COUNT` and `CUBE_POS` variables, which will
allow us to launch multiple invocation in parallel without exceeding the limits of the `CUBE_DIM`.
Launch one cube per first-dimension index and one unit per second-dimension index. `CUBE_POS_X` and
`UNIT_POS_X` identify the row, and the output shape is the first two input dimensions. The cube size
must still fit the device limits.

```rust,ignore
{{#include src/bin/v7-gpu.rs:implementation}}
```

The kernel keeps the same accumulation loop as the vectorized 2D example. Only the row coordinates
change: v6 uses two unit coordinates, while v7 uses a cube coordinate and a unit coordinate. Tensor
strides count scalar elements, so the input offset is divided by `N::value()` to index vectors.
`acc.vector_sum()` combines the vector lanes into the single scalar result for each row; writing
`acc` directly would leave separate partial sums. The output is a scalar tensor with shape
`[input_shape[0], input_shape[1]]`.

Now, let's run the code again.

```
wgpu<wgsl>-reduction-[64, 256, 1024]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        1.483ms
  Variance    27.000ns
  Median      1.535ms
  Min         1.239ms
  Max         1.808ms
―――――――――――――――――――――――――
wgpu<wgsl>-reduction-[64, 64, 4096]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        924.409µs
  Variance    189.000ns
  Median      945.270µs
  Min         600.110µs
  Max         2.098ms
―――――――――――――――――――――――――
```

It runs and it is fast! The reduction operation is now parallelized across multiple invocations, and
we can see that the performance is significantly improved compared to the previous implementation.
The results show that the reduction operation is efficient and can handle larger tensors without
exceeding the GPU limits. It's also almost the same speed as the 2D reduction, even if there's even
more elements to reduce. This is because the reduction is now parallelized across multiple cubes and
hyper-cubes, allowing the GPU to process the data more efficiently. See the
[parallel reduction](../getting-started/parallel_reduction.md) if you need a refresher on the
different parallelization level used in CubeCL. It is also worth noting that the performance and
optimal `CUBE_COUNT` and `CUBE_DIM` values may vary depending on the GPU architecture and the
specific workload. You may need to experiment with different values to find the best configuration
for your use case.
