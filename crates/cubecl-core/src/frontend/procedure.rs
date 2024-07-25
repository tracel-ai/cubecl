use crate::{frontend::CubeContext, unexpanded};

use super::{CubePrimitive, Tensor, UInt};

pub fn index_offset_global_with_layout<T: CubePrimitive>(
    _tensors: Vec<&Tensor<T>>,
    _indexes: Vec<UInt>,
    _layout: &Tensor<T>,
    _position: UInt,
    _dim_start: UInt,
    _dim_end: UInt,
) {
    unexpanded!()
}

pub mod index_offset_global_with_layout {
    use crate::{ir::IndexOffsetGlobalWithLayout, prelude::ExpandElementTyped};

    use super::*;

    pub fn __expand<T: CubePrimitive>(
        context: &mut CubeContext,
        tensors: Vec<ExpandElementTyped<Tensor<T>>>,
        indexes: Vec<ExpandElementTyped<UInt>>,
        layout: ExpandElementTyped<Tensor<T>>,
        position: ExpandElementTyped<UInt>,
        dim_start: ExpandElementTyped<UInt>,
        dim_end: ExpandElementTyped<UInt>,
    ) {
        IndexOffsetGlobalWithLayout {
            tensors: tensors.into_iter().map(|t| t.expand.into()).collect(),
            indexes: indexes.into_iter().map(|t| t.expand.into()).collect(),
            layout: layout.expand.into(),
            position: position.expand.into(),
            dim_start: dim_start.expand.into(),
            dim_end: dim_end.expand.into(),
        }
        .expand(&mut context.child().into_scope())
    }
}
