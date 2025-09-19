use cubecl_runtime::stride::contiguous_strides;

/// Example of a naive multidimensional tensor in pure Rust
#[derive(Debug, Clone)]
pub struct CpuTensor {
    /// Raw contiguous value buffer
    pub data: Vec<f32>,
    /// How many element are between each dimensions
    pub strides: Vec<usize>,
    /// Dimension of the tensor
    pub shape: Vec<usize>,
}


impl CpuTensor {
    /// Create a CpuTensor with a shape filled by number in order
    pub fn arange(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = (0..size).map(|i| i as f32).collect();
        let strides = contiguous_strides(&shape);
        Self {
            data,
            strides,
            shape,
        }
    }

    /// Create an empty CpuTensor with a shape
    pub fn empty(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![0.0; size];
        let strides = contiguous_strides(&shape);
        Self {
            data,
            strides,
            shape,
        }
    }

    /// Read the inner data
    pub fn read(self) -> Vec<f32> {
        self.data
    }
}
