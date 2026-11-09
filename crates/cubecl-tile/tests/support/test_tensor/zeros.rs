use cubecl::{std::tensor::TensorHandle, zspace::metadata::Metadata};

use crate::support::BaseInputSpec;

pub(crate) fn build_zeros(spec: BaseInputSpec) -> TensorHandle {
    let mut tensor = TensorHandle::zeros(&spec.client, spec.shape.clone(), spec.dtype);

    // This manipulation is only valid since all the data is the same
    *tensor.metadata = Metadata::new(tensor.shape(), spec.strides());

    tensor
}
