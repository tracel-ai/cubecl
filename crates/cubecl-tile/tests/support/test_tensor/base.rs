use cubecl::{
    ir::ElemType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::support::test_tensor::{
    arange::build_arange,
    custom::build_custom,
    host_data::{HostData, HostDataType},
    random::build_uniform,
    strides::StridedLayout,
    zeros::build_zeros,
};

/// A device tensor a test is about to create: where, of what shape and layout, and its values.
pub struct TestInput {
    spec: BaseInputSpec,
    data: DataKind,
}

enum DataKind {
    Arange,
    Zeros,
    Uniform { seed: u64, lo: f32, hi: f32 },
    Custom(Vec<f32>),
}

impl TestInput {
    /// Start a builder: `f32`, row-major, until `.dtype(_)` / `.layout(_)` say otherwise; a
    /// finalizer (`arange`, `zeros`, `uniform`, `custom`) names the values.
    pub fn builder(client: Client, shape: impl Into<Shape>) -> TestInputBuilder {
        TestInputBuilder {
            client,
            shape: shape.into(),
            dtype: f32::elem_type_native(),
            layout: StridedLayout::RowMajor,
        }
    }

    pub fn generate_with_f32_host_data(self) -> (TensorHandle, HostData) {
        let client = self.spec.client.clone();
        let handle = self.generate_without_host_data();
        let host = HostData::from_tensor_handle(&client, handle.clone(), HostDataType::F32);
        (handle, host)
    }

    pub fn generate_without_host_data(self) -> TensorHandle {
        let (shape, strides, dtype) = (
            self.spec.shape.clone(),
            self.spec.strides(),
            self.spec.dtype,
        );

        let mut handle = match self.data {
            DataKind::Arange => build_arange(self.spec),
            DataKind::Zeros => build_zeros(self.spec),
            DataKind::Uniform { seed, lo, hi } => build_uniform(self.spec, seed, lo, hi),
            DataKind::Custom(data) => build_custom(self.spec, data),
        };
        handle.metadata.shape = shape;
        handle.metadata.strides = strides;
        handle.dtype = dtype;

        handle
    }
}

pub struct BaseInputSpec {
    pub client: Client,
    pub shape: Shape,
    pub dtype: ElemType,
    pub layout: StridedLayout,
}

impl BaseInputSpec {
    pub(crate) fn strides(&self) -> Strides {
        self.layout.compute_strides(&self.shape)
    }
}

/// Builder for a [`TestInput`]; start one with [`TestInput::builder`].
pub struct TestInputBuilder {
    client: Client,
    shape: Shape,
    dtype: ElemType,
    layout: StridedLayout,
}

impl TestInputBuilder {
    pub fn dtype(mut self, dtype: ElemType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn layout(mut self, layout: StridedLayout) -> Self {
        self.layout = layout;
        self
    }

    fn finalize(self, data: DataKind) -> TestInput {
        let spec = BaseInputSpec {
            client: self.client,
            shape: self.shape,
            dtype: self.dtype,
            layout: self.layout,
        };
        TestInput { spec, data }
    }

    /// `0, 1, 2, …` in row-major order.
    pub fn arange(self) -> TestInput {
        self.finalize(DataKind::Arange)
    }

    pub fn zeros(self) -> TestInput {
        self.finalize(DataKind::Zeros)
    }

    /// Uniform random in `[lo, hi]`, reproducible from `seed`.
    pub fn uniform(self, seed: u64, lo: f32, hi: f32) -> TestInput {
        self.finalize(DataKind::Uniform { seed, lo, hi })
    }

    /// The values of `data`, in row-major order.
    pub fn custom(self, data: Vec<f32>) -> TestInput {
        self.finalize(DataKind::Custom(data))
    }
}
