# Simple Reduction

To get started with CubeCL, we will implement a simple reduction operation on a multidimensional
array (tensor). This example will help you understand the basic concepts of CubeCL and how to use it
to perform parallel computations on tensors.

## An example of a CPU reduction in Rust without CubeCL

This example demonstrates how to perform a simple reduction operation on a multidimensional array
(tensor) using pure Rust. The code is designed to be easy to understand and serves as a starting
point for more complex operations that can be parallelized with CubeCL. It is not optimized for
performance, but it illustrates the basic concepts of working with tensors and performing
reductions.

### CpuTensor Struct

Tensors are the basic data structure used in CubeCL to represent multidimensional arrays. Here is a
simple implementation of a tensor in pure Rust. It is not optimized for performance, but it is easy
to understand.

```rust,ignore
{{#include src/cpu_tensor.rs}}
```

### Reduce function

The following function is a naive implementation of a reduction operation on a matrix. It sums the
values of each row and stores the result in a new tensor. The input tensor is expected to be a 2D
matrix, and the output tensor will be a 1D vector containing the sum of each row.

```rust,ignore
{{#include src/bin/v1-cpu.rs:implementation}}
```

### Launching code

The following code creates a 3x3 matrix, initializes the input tensor, and calls the `reduce_matrix`
function to perform the reduction. The result is printed to the console.

```rust,ignore
{{#include src/bin/v1-cpu.rs:launch}}
```

## A first example of a GPU reduction with CubeCL

This example demonstrates how to perform a simple reduction operation on a multidimensional array
(tensor) using CubeCL. It is a simple implementation that will be used as a starting point to show
how to use CubeCL in the next chapters.

## GpuTensor struct

The `GpuTensor<F>` struct stores a device buffer and its shape. It is generic over the
floating-point type, while the `Client` manages the selected device at runtime. Its `as_arg` method
creates a `TensorArg` carrying the buffer, shape, and compact strides. The kernel reads this
metadata through `shape` and `stride`.

A `Client` allocates buffers, launches kernels, and reads results from the device. Obtain it with
`device.client()`.

<div class="warning">
If you need a tensor library instead of defining your own kernel and tensor, you should use <a href=https://github.com/tracel-ai/burn target="_blank">Burn</a> directly instead.
</div>

```rust,ignore
{{#include src/gpu_tensor.rs}}
```

## Reduce function

Compared to the previous example, this function is similar but uses CubeCL's `cube` macro to define
the kernel. The kernel performs the same reduction operation, summing the values of each row and
storing the result in a new tensor. The variable `F` is a generic type that implements the `Float`
trait, allowing the function to work with different floating-point types (e.g., `f32`, `f64`). The
kernel takes an input `&Tensor<F>` and an output `&mut Tensor<F>`. Tensor arguments provide the
shape and stride metadata needed for the reduction without extra parameters.

```rust,ignore
{{#include src/bin/v2-gpu.rs:implementation}}
```

### Launching code

Once the kernel is defined, we can launch it using CubeCL's runtime. The following code creates a
3x3 matrix, initializes the input tensor, and calls the `reduce_matrix` function to perform the
reduction. The result is printed to the console. The example uses `Device::default()` to select a
device from the enabled runtimes. With the `wgpu` feature enabled, this can run on WebGPU.

```rust,ignore
{{#include src/bin/v2-gpu.rs:launch}}
```
