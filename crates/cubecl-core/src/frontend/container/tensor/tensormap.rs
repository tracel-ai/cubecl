use std::marker::PhantomData;

use crate::prelude::*;
use crate::{
    ir::{ExpandElement, Item},
    ConstantInfo,
};
use cubecl_common::{
    OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle,
};
use cubecl_ir::{Elem, Variable, VariableKind};
use serde::{Deserialize, Serialize};

/// Grid constant tensor map, currently only maps to CUDA tensormap. May be interleaved or swizzled,
/// but last dimension must be contiguous (since strides don't include the last dimension).
///
/// The tensormap is treated as an opaque type at runtime.
pub struct TensorMapArg<'a, R: Runtime> {
    pub format: TensorMapFormat,
    pub tensor: TensorArg<'a, R>,
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    pub elem_stride: Vec<u32>,
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub prefetch: TensorMapPrefetch,
    pub oob_fill: OobFill,
    pub elem: Elem,
}

impl<'a, R: Runtime> TensorMapArg<'a, R> {
    pub fn new(format: TensorMapFormat, tensor: TensorArg<'a, R>, rank: usize, elem: Elem) -> Self {
        Self {
            format,
            tensor,
            elem_stride: vec![1; rank],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: TensorMapPrefetch::None,
            oob_fill: OobFill::Zero,
            elem,
        }
    }

    pub fn with_elem_stride(mut self, elem_stride: Vec<u32>) -> Self {
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

#[derive(Clone)]
pub struct TensorMap<E: CubePrimitive> {
    _ty: PhantomData<E>,
}

impl<E: CubePrimitive> Copy for TensorMap<E> {}

impl<E: CubePrimitive> TensorMap<E> {
    pub fn dummy() -> Self {
        TensorMap { _ty: PhantomData }
    }

    pub fn __expand_dummy(_scope: &mut Scope) -> ExpandElementTyped<Self> {
        let x: ExpandElement = 0.into();
        x.into()
    }
}

impl<E: CubePrimitive> IntoRuntime for TensorMap<E> {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
        ExpandElementTyped {
            expand: ExpandElement::Plain(Variable::new(
                VariableKind::TensorMap(0),
                Item::new(E::as_elem_native_unchecked()),
            )),
            _type: PhantomData,
        }
    }
}

impl<E: CubePrimitive> ExpandElementBaseInit for TensorMap<E> {
    fn init_elem(_scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        elem
    }
}

impl<E: CubePrimitive> CubeType for TensorMap<E> {
    type ExpandType = ExpandElementTyped<TensorMap<E>>;
}

impl<E: CubePrimitive> CubeType for *const TensorMap<E> {
    type ExpandType = ExpandElementTyped<TensorMap<E>>;
}

impl<E: CubePrimitive> CubeType for *mut TensorMap<E> {
    type ExpandType = ExpandElementTyped<TensorMap<E>>;
}

impl<R: Runtime> ArgSettings<R> for TensorMapArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor_map(self)
    }
}

/// Compilation argument for a [tensor](Tensor).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TensorMapCompilationArg;

impl CompilationArg for TensorMapCompilationArg {}

impl<E: CubePrimitive> LaunchArgExpand for TensorMap<E> {
    type CompilationArg = TensorMapCompilationArg;

    fn expand(
        _arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E>> {
        let tensor = builder.constant(ConstantInfo::TensorMap);
        tensor.into()
    }
    fn expand_output(
        _arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E>> {
        let tensor = builder.constant(ConstantInfo::TensorMap);
        tensor.into()
    }
}

impl<E: CubePrimitive> LaunchArg for TensorMap<E> {
    type RuntimeArg<'a, R: Runtime> = TensorMapArg<'a, R>;

    fn compilation_arg<R: Runtime>(_runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        TensorMapCompilationArg
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CopyDirection {
    GlobalToShared,
    SharedToGlobal,
}
