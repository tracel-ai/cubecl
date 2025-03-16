use std::marker::PhantomData;

use crate::ir::{ExpandElement, Item};
use crate::prelude::*;
use cubecl_common::{
    OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle,
};
use cubecl_ir::Elem;
use serde::{Deserialize, Serialize};

/// Grid constant tensor map, currently only maps to CUDA tensormap. May be interleaved or swizzled,
/// but last dimension must be contiguous (since strides don't include the last dimension).
///
/// The tensormap is treated as an opaque type at runtime.
pub struct TensorMapArg<'a, R: Runtime, const RANK: usize> {
    pub format: TensorMapFormat,
    pub tensor: TensorArg<'a, R>,
    // The `shared_shape` is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    pub shared_shape: [u32; RANK],
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    pub elem_stride: [u32; RANK],
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub prefetch: TensorMapPrefetch,
    pub oob_fill: OobFill,
    pub elem: Elem,
}

impl<'a, R: Runtime, const RANK: usize> TensorMapArg<'a, R, RANK> {
    pub fn new(
        format: TensorMapFormat,
        tensor: TensorArg<'a, R>,
        shared_shape: [u32; RANK],
        elem: Elem,
    ) -> Self {
        Self {
            format,
            tensor,
            shared_shape,
            elem_stride: [1; RANK],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: TensorMapPrefetch::None,
            oob_fill: OobFill::Zero,
            elem,
        }
    }

    pub fn with_elem_stride(mut self, elem_stride: [u32; RANK]) -> Self {
        self.elem_stride = elem_stride;
        self
    }

    pub fn with_interleave(mut self, interleave: TensorMapInterleave) -> Self {
        self.interleave = interleave;
        self
    }

    pub fn with_swizzle(mut self, swizzle: TensorMapSwizzle) -> Self {
        self.swizzle = swizzle;
        self
    }

    pub fn with_prefetch(mut self, prefetch: TensorMapPrefetch) -> Self {
        self.prefetch = prefetch;
        self
    }

    pub fn with_nan_fill(mut self) -> Self {
        self.oob_fill = OobFill::NaN;
        self
    }
}

pub struct TensorMap<E: CubePrimitive, const RANK: usize> {
    _ty: PhantomData<E>,
}

impl<E: CubePrimitive, const RANK: usize> ExpandElementBaseInit for TensorMap<E, RANK> {
    fn init_elem(_scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        elem
    }
}

impl<E: CubePrimitive, const RANK: usize> CubeType for TensorMap<E, RANK> {
    type ExpandType = ExpandElementTyped<TensorMap<E, RANK>>;
}

impl<R: Runtime, const RANK: usize> ArgSettings<R> for TensorMapArg<'_, R, RANK> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor(&self.tensor)
    }
}

/// Compilation argument for a [tensor](Tensor).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TensorMapCompilationArg {
    pub tensor: TensorCompilationArg,
    pub rank: usize,
}

impl CompilationArg for TensorMapCompilationArg {}

impl<E: CubePrimitive, const RANK: usize> LaunchArgExpand for TensorMap<E, RANK> {
    type CompilationArg = TensorMapCompilationArg;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E, RANK>> {
        let tensor = builder.input_tensor(Item::vectorized(
            E::as_elem(&builder.context),
            arg.tensor.vectorisation,
        ));
        tensor.into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E, RANK>> {
        match arg.tensor.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder
                .output_tensor(Item::vectorized(
                    E::as_elem(&builder.context),
                    arg.tensor.vectorisation,
                ))
                .into(),
        }
    }
}

impl<E: CubePrimitive, const RANK: usize> LaunchArg for TensorMap<E, RANK> {
    type RuntimeArg<'a, R: Runtime> = TensorMapArg<'a, R, RANK>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        TensorMapCompilationArg {
            tensor: <Tensor<E> as LaunchArg>::compilation_arg(&runtime_arg.tensor),
            rank: RANK,
        }
    }
}
