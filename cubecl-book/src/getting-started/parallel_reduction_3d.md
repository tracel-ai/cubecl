# Parallel reduction 3D
The purpose of this example is to demonstrate how to perform a parallel reduction operation on a 3D tensor using CubeCL. The reduction will sum the elements along the last dimension (depth) of the tensor, resulting in a 2D tensor.

## A first try
We will start with a simple implementation of a parallel reduction on a 3D tensor. The goal is to reduce the tensor along the last dimension (depth) by summing the elements. This will result in a 2D tensor where each element is the sum of the corresponding elements in the depth dimension.

```rust,ignore
{{#rustdoc_include src/bin/v6-gpu.rs:31:57}}
```

Let's try to run this code.
```
wgpu error: Validation Error

Caused by:
  In Device::create_compute_pipeline, label = 'reduce_matrix_f32'
    Error matching shader requirements against the pipeline
      Shader entry point's workgroup size [64, 256, 1] (16384 total invocations) must be less or equal to the per-dimension limit [1024, 1024, 1024] and the total invocation limit 1024
```

What happened? The error message indicates that the workgroup size exceeds the limits imposed by the WebGPU backend. The total number of invocations (64 * 256 * 1 = 16384) exceeds the maximum allowed invocations per workgroup, which is 1024. In other words, the CubeDim size is too large for the GPU to handle. We needs to find another way to parallelize the reduction operation without exceeding the limits.

## A better approach
To address the issue, we will parallelize with the `CUBE_COUNT` and `CUBE_POS` variables, which will allow us to launch multiple invocation in parallel without exceeding the limits of the `CUBE_DIM`. The `CUBE_COUNT` variable will determine how many invocation we will launch, and the `CUBE_POS` variable will determine the position of each invocation in the 3D tensor.

```rust,ignore
{{#rustdoc_include src/bin/v7-gpu.rs:31:57}}
```
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
It runs and it is fast! The reduction operation is now parallelized across multiple invocations, and we can seeb that the performance is significantly improved compared to the previous implementation. The results show that the reduction operation is efficient and can handle larger tensors without exceeding the GPU limits. It's also almost the same speed as the 2D reduction, even if there's even more elements to reduce. This is because the reduction is now parallelized across multiple cubes and hyper-cubes, allowing the GPU to process the data more efficiently. See the [parallel reduction](../getting-started/parallel_reduction.md) if you need a refresher on the different parallelization level used in CubeCL. It is also worth noting that the performance and optimal `CUBE_COUNT` and `CUBE_DIM` values may vary depending on the GPU architecture and the specific workload. You may need to experiment with different values to find the best configuration for your use case.
