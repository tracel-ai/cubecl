#![allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#![allow(clippy::manual_div_ceil)]

/// Contains matmul kernels and Cube components
pub mod matmul;

/// Contains convolution using matmul components
pub mod convolution;

/// Contains basic tensor helpers.
pub mod tensor;
