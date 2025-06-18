# Simple Reduction

To get started with CubeCL, we will implement a simple reduction operation on a multidimensional array (tensor). This example will help you understand the basic concepts of CubeCL and how to use it to perform parallel computations on tensors.

## An example of a CPU reduction in Rust without CubeCL
This example demonstrates how to perform a simple reduction operation on a multidimensional array (tensor) using pure Rust. The code is designed to be easy to understand and serves as a starting point for more complex operations that can be parallelized with CubeCL. It is not optimized for performance, but it illustrates the basic concepts of working with tensors and performing reductions.

### Tensor Struct
Tensor are the basic data structure used in CubeCL to represent multidimensional arrays. Here is a simple implementation of a tensor in pure Rust. It is not optimized for performance, but it is easy to understand.
```rust,ignore
/// Example of a naive multidimensional tensor in pure Rust
pub struct Tensor {
    /// Raw contiguous value buffer
    values: Vec<f32>,
    /// How many element are between each dimension
    stride: Vec<usize>,
    /// Dimension of the tensor
    shape: Vec<usize>,
}
```

### Reduce function
The following function is a naive implementation of a reduction operation on a matrix. It sums the values of each row and stores the result in a new tensor. The input tensor is expected to be a 2D matrix, and the output tensor will be a 1D vector containing the sum of each row.
```rust,ignore
# /// Example of a naive multidimensional tensor in pure Rust
# pub struct Tensor {
#     /// Raw contiguous value buffer
#     values: Vec<f32>,
#     /// How many element are between each dimension
#     stride: Vec<usize>,
#     /// Dimension of the tensor
#     shape: Vec<usize>,
# }

/// This function execute the reduction in the following way by reducing with a sum
/// [0 1 2]    [0 + 1 + 2]    [3 ]
/// [3 4 5] -> [3 + 4 + 5] -> [12]
/// [6 7 8]    [6 + 7 + 8]    [21]
fn reduce_matrix(input: &Tensor, output: &mut Tensor) {
    for i in 0..input.shape[0] {
        let mut acc = 0.0f32;
        for j in 0..input.shape[1] {
            acc += input.values[i * input.stride[0] + j];
        }
        output.values[i] = acc;
    }
}
```

### Launching code
The following code creates a 3x3 matrix, initializes the input tensor, and calls the `reduce_matrix` function to perform the reduction. The result is printed to the console.
```rust
# /// Example of a naive multidimensional tensor in pure Rust
# pub struct Tensor {
#     /// Raw contiguous value buffer
#     values: Vec<f32>,
#     /// How many element are between each dimension
#     stride: Vec<usize>,
#     /// Dimension of the tensor
#     shape: Vec<usize>,
# }
#
# /// This function execute the reduction in the following way by reducing with a sum
# /// [0 1 2]    [0 + 1 + 2]    [3 ]
# /// [3 4 5] -> [3 + 4 + 5] -> [12]
# /// [6 7 8]    [6 + 7 + 8]    [21]
# fn reduce_matrix(input: &Tensor, output: &mut Tensor) {
#     for i in 0..input.shape[0] {
#         let mut acc = 0.0f32;
#         for j in 0..input.shape[1] {
#             acc += input.values[i * input.stride[0] + j];
#         }
#         output.values[i] = acc;
#     }
# }

fn launch() {
    let shape = vec![3, 3];
    let stride = vec![shape[1], 1];
    // create a matrix with [0, 1, 2, 3, 4, 5, 6, 7, 8]
    let values: Vec<f32> = (0..shape[0] * shape[1])
        .into_iter()
        .map(|i| i as f32)
        .collect();
    let input = Tensor {
        values,
        stride,
        shape,
    };

    let shape = vec![input.shape[0]];
    let stride = vec![1];
    let values = vec![0.0; input.shape[0]];
    let mut output = Tensor {
        values,
        stride,
        shape,
    };

    reduce_matrix(&input, &mut output);
    println!("Executed reduction => {:?}", output.values);
}

fn main() {
    launch();
}
```

## A first example of a GPU reduction with CubeCL
This example demonstrates how to perform a simple reduction operation on a multidimensional array (tensor) using CubeCL. It is a simple implementation that will be used as a starting point to show how to use CubeCL in the next chapters.

## Reduce function
Compared to the previous example, this function is similar but uses CubeCL's `cube` macro to define the kernel. The kernel performs the same reduction operation, summing the values of each row and storing the result in a new tensor. The variable `F` is a generic type that implements the `Float` trait, allowing the function to work with different floating-point types (e.g., `f32`, `f64`). The tensor is provided by cubecl::prelude, which includes the necessary traits and types for using CubeCL.
```rust
use cubecl::prelude::*;

#[cube(launch_unchecked)]
/// This function execute the reduction in the following way by reducing with a sum
/// [0 1 2]    [0 + 1 + 2]    [3 ]
/// [3 4 5] -> [3 + 4 + 5] -> [12]
/// [6 7 8]    [6 + 7 + 8]    [21]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    for i in 0..input.shape(0) {
        let mut acc = F::new(0.0f32);
        for j in 0..input.shape(1) {
            acc += input[i * input.stride(0) + j];
        }
        output[i] = acc;
    }
}
```

### Launching code
Once the kernel is defined, we can launch it using CubeCL's runtime. The following code creates a 3x3 matrix, initializes the input tensor, and calls the `reduce_matrix` function to perform the reduction. The result is printed to the console. Note that this code uses the `cubecl::wgpu::WgpuRuntime` runtime, which is a CubeCL runtime for WebGPU. You can replace it with any other CubeCL runtime that you prefer. 
```rust
pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let shapes = [3, 3];
    let strides = [shapes[1], 1];

    let input: Vec<f32> = (0..shapes[0] * shapes[1])
        .into_iter()
        .map(|i| i as f32)
        .collect();

    let output_handle = client.empty(input.len() / shapes[1] * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(&input));

    unsafe {
        reduce_matrix::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<f32>(&input_handle, &strides, &shapes, 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &strides, &[shapes[0]], 1),
        )
    };

    let bytes = client.read_one(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed reduction with runtime {:?} => {output:?}",
        R::name(&client)
    );
}

fn main() {
    launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
```
