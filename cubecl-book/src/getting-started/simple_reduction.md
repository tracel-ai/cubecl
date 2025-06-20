# Simple Reduction

To get started with CubeCL, we will implement a simple reduction operation on a multidimensional array (tensor). This example will help you understand the basic concepts of CubeCL and how to use it to perform parallel computations on tensors.

## An example of a CPU reduction in Rust without CubeCL
This example demonstrates how to perform a simple reduction operation on a multidimensional array (tensor) using pure Rust. The code is designed to be easy to understand and serves as a starting point for more complex operations that can be parallelized with CubeCL. It is not optimized for performance, but it illustrates the basic concepts of working with tensors and performing reductions.

### CpuTensor Struct
Tensors are the basic data structure used in CubeCL to represent multidimensional arrays. Here is a simple implementation of a tensor in pure Rust. It is not optimized for performance, but it is easy to understand.
```rust,ignore
{{#include src/cpu_tensor.rs}}
```

### Reduce function
The following function is a naive implementation of a reduction operation on a matrix. It sums the values of each row and stores the result in a new tensor. The input tensor is expected to be a 2D matrix, and the output tensor will be a 1D vector containing the sum of each row.
```rust,ignore
{{#include src/bin/v1-cpu.rs:1:15}}
```

### Launching code
The following code creates a 3x3 matrix, initializes the input tensor, and calls the `reduce_matrix` function to perform the reduction. The result is printed to the console.
```rust,ignore
{{#rustdoc_include src/bin/v1-cpu.rs:17:31}}
```

## A first example of a GPU reduction with CubeCL
This example demonstrates how to perform a simple reduction operation on a multidimensional array (tensor) using CubeCL. It is a simple implementation that will be used as a starting point to show how to use CubeCL in the next chapters.

## GpuTensor struct
The `GpuTensor` struct is a representation of a tensor that resides on the GPU. It contains the data handle, shape, strides, and marker types for the runtime and floating-point type. The `GpuTensor` struct provides methods to create tensors, read data from the GPU, and convert them into tensor arguments for kernel execution. Please note that it is generic over the runtime and floating-point type, allowing it to work with different CubeCL runtimes and floating-point types (e.g., `f16`, `f32`). Also, the strides can be computed using the `compact_strides` function from the `cubecl::std::tensor` module, which will compute the strides for a given shape with a compact representation.

Another important concept is the `ComputeClient` trait, which define what a runtime should implement to be able to run kernels. Each runtime has their own implementation of the `ComputeClient` trait, which provides methods to create tensors and read data from the GPU. The `ComputeClient` can send compute task to a `Server` that will run the kernel on the GPU and schedule the tasks.

```rust,ignore
{{#include src/gpu_tensor.rs:0:1}}
```

<div class="warning">
If you need a tensor library instead of defining your own kernel and tensor, you should use <a href=https://github.com/tracel-ai/burn target="_blank">Burn</a> directly instead.
</div>

```rust,ignore
{{#include src/gpu_tensor.rs}}
```

## Reduce function
Compared to the previous example, this function is similar but uses CubeCL's `cube` macro to define the kernel. The kernel performs the same reduction operation, summing the values of each row and storing the result in a new tensor. The variable `F` is a generic type that implements the `Float` trait, allowing the function to work with different floating-point types (e.g., `f32`, `f64`). The tensor is provided by cubecl::prelude, which includes the necessary traits and types for using CubeCL.
```rust,ignore
{{#include src/bin/v2-gpu.rs:1:13}}
```

### Launching code
Once the kernel is defined, we can launch it using CubeCL's runtime. The following code creates a 3x3 matrix, initializes the input tensor, and calls the `reduce_matrix` function to perform the reduction. The result is printed to the console. Note that this code uses the `cubecl::wgpu::WgpuRuntime` runtime, which is a CubeCL runtime for WebGPU. You can replace it with any other CubeCL runtime that you prefer.
```rust,ignore
{{#rustdoc_include src/bin/v2-gpu.rs:15:}}
```
