# Vectorized Reduction
In this section, we will explore how to implement a vectorized reduction operation using CubeCL. Vectorization is a powerful technique that allows us to process multiple data elements simultaneously, significantly improving performance for certain types of computations especially I/O operations.

## What is vectorization?
Vectorization is the process of converting scalar operations (which operate on single data elements) into vector operations (which operate on multiple data elements simultaneously). This is typically done using SIMD (Single Instruction, Multiple Data) instructions available in modern CPUs and GPUs. By leveraging vectorization, we can achieve significant performance improvements for operations that can be vectorized. For more information on vectorization in CubeCL, you can refer to [this section](../core-features/vectorization.md).

## Application to the reduction problem
To apply vectorization to the reduction problem, we will modify our reduction kernel to process multiple elements at once. This means that instead of summing one element at a time, we will sum multiple elements with vectorization, which can lead to substantial performance gains. The number of element processed at a time is the line size. So to add vectorization we just needs to pass the `LINE_SIZE` to the `TensorArgs` and reduce the number of iteration of the `reduce_matrix`.

```rust,ignore
{{#rustdoc_include src/bin/v5-gpu.rs:13:57}}
```

## The Result
The result of adding vectorization is an average of 3x speedup compared to the previous parallel reduction implementation. This is because we are now processing multiple elements at a time in each invocation, which reduces the time of running a single invocation.
```
wgpu<wgsl>-reduction-[512, 8192]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        1.085ms
  Variance    14.000ns
  Median      1.045ms
  Min         998.981µs
  Max         1.375ms
―――――――――――――――――――――――――
wgpu<wgsl>-reduction-[128, 32768]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        3.124ms
  Variance    37.000ns
  Median      3.061ms
  Min         3.009ms
  Max         3.670ms
―――――――――――――――――――――――――
```
