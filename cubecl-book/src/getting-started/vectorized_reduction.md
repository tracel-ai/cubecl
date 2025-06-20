# Vectorized Reduction
In this section, we will explore how to implement a vectorized reduction operation using CubeCL. Vectorization is a powerful technique that allows us to process multiple data elements simultaneously, significantly improving performance for certain types of computations.

## What is vectorization?
Vectorization is the process of converting scalar operations (which operate on single data elements) into vector operations (which operate on multiple data elements simultaneously). This is typically done using SIMD (Single Instruction, Multiple Data) instructions available in modern CPUs and GPUs. By leveraging vectorization, we can achieve significant performance improvements for operations that can be vectorized. For more information on vectorization in CubeCL, you can refer to [this section](../core-features/vectorization.md).

## Application to the reduction problem
To apply vectorization to the reduction problem, we will modify our reduction kernel to process multiple elements at once. This means that instead of summing one element at a time, we will sum multiple elements with vectorization, which can lead to substantial performance gains. This is done by passing the `VECTORIZATION_FACTOR` to the `TensorArgs` and reducing the number of iteration of the `reduce_matrix`.

```rust,ignore
{{#rustdoc_include code_example/bin/v5-gpu.rs:13:57}}
```
