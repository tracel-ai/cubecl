use cubecl::{
    CubeElement,
    client::Client,
    frontend::Scalar,
    ir::{ElemType, FloatKind},
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::support::test_tensor::{cast::copy_casted, strides::physical_extent};

#[derive(Debug, Clone)]
pub struct HostData {
    pub data: HostDataVec,
    pub shape: Shape,
    pub strides: Strides,
}

#[derive(Eq, PartialEq, PartialOrd, Clone, Copy, Debug)]
pub enum HostDataType {
    F32,
    F64,
    I32,
}

/// Convert a `Vec<f64>` accumulation buffer into a `HostDataVec` at the
/// precision dictated by `dtype`: kept as `F64` for f64 tensors, downcast to
/// `F32` for everything else (matching `HostDataType::from`).
impl From<(Vec<f64>, ElemType)> for HostDataVec {
    fn from((data, dtype): (Vec<f64>, ElemType)) -> Self {
        match HostDataType::from(dtype) {
            HostDataType::F64 => HostDataVec::F64(data),
            _ => HostDataVec::F32(data.into_iter().map(|x| x as f32).collect()),
        }
    }
}

impl From<ElemType> for HostDataType {
    /// Map a tensor storage type to the host representation used in tests.
    /// `f64` tensors get `F64` so comparisons happen at full precision; all
    /// other numeric types collapse to `F32`.
    fn from(dtype: ElemType) -> Self {
        if matches!(dtype, ElemType::Float(FloatKind::F64)) {
            HostDataType::F64
        } else {
            HostDataType::F32
        }
    }
}

#[derive(Clone, Debug)]
pub enum HostDataVec {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
}

impl HostDataVec {
    pub fn get_f32(&self, i: usize) -> f32 {
        match self {
            HostDataVec::F32(items) => items[i],
            _ => panic!("Can't get as f32"),
        }
    }

    /// Get the element as `f64`, losslessly upcasting `F32` data if needed.
    /// This is the precision-preserving accessor used by the correctness
    /// comparator so `F64` host data can be compared at full precision.
    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            HostDataVec::F32(items) => items[i] as f64,
            HostDataVec::F64(items) => items[i],
            _ => panic!("Can't get as f64"),
        }
    }

    pub fn get_i32(&self, i: usize) -> i32 {
        match self {
            HostDataVec::I32(items) => items[i],
            _ => panic!("Can't get as i32"),
        }
    }
}

impl HostData {
    pub fn from_tensor_handle(
        client: &Client,
        mut tensor_handle: TensorHandle,
        host_data_type: HostDataType,
    ) -> Self {
        let shape = tensor_handle.shape().clone();
        let strides = tensor_handle.strides().clone();

        // Reshape to a flat 1D view of the full physical buffer so the read
        // covers every offset the jumpy strides might reach. Without this, a
        // shape like [256,256] with strides [512,1] would only read the
        // shape.product() (65536) elements that `copy_casted`'s contiguous
        // rewrite walks, and HostData.get_f32 would then index out-of-bounds
        // when the logical walk crosses the padding.
        let physical_len = physical_extent(&shape, &strides);
        tensor_handle.metadata.shape = Shape::from(vec![physical_len]);
        tensor_handle.metadata.strides = Strides::new(&[1]);

        let data = match host_data_type {
            HostDataType::F32 => {
                let handle = copy_casted(client, tensor_handle, f32::elem_type_native());
                let data = f32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::F32(data)
            }
            HostDataType::F64 => {
                let handle = copy_casted(client, tensor_handle, f64::elem_type_native());
                let data = f64::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::F64(data)
            }
            HostDataType::I32 => {
                let handle = copy_casted(client, tensor_handle, i32::elem_type_native());
                let data = i32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::I32(data)
            }
        };

        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn get_f32(&self, index: &[usize]) -> f32 {
        self.data.get_f32(self.strided_index(index))
    }

    /// Like [`get_f32`](Self::get_f32), but returns `f64`, losslessly
    /// upcasting `F32` data if needed.
    pub fn get_f64(&self, index: &[usize]) -> f64 {
        self.data.get_f64(self.strided_index(index))
    }

    pub fn get_i32(&self, index: &[usize]) -> i32 {
        self.data.get_i32(self.strided_index(index))
    }

    fn strided_index(&self, index: &[usize]) -> usize {
        let mut i = 0usize;
        for (d, idx) in index.iter().enumerate() {
            i += idx * self.strides[d];
        }
        i
    }
}
