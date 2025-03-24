use std::marker::PhantomData;

use crate::{ConstantInfo, ir::ExpandElement};
use crate::{prelude::*, unexpanded};
use cubecl_common::{
    OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle,
};
use cubecl_ir::Elem;
use cubecl_runtime::server::TensorMapMeta;
use serde::{Deserialize, Serialize};

/// Grid constant tensor map, currently only maps to CUDA tensormap. May be interleaved or swizzled,
/// but last dimension must be contiguous (since strides don't include the last dimension).
///
/// The tensormap is treated as an opaque type at runtime.
///
pub struct TensorMapArg<'a, R: Runtime> {
    pub(crate) tensor: TensorArg<'a, R>,
    pub(crate) metadata: TensorMapMeta,
}

impl<'a, R: Runtime> TensorMapArg<'a, R> {
    pub fn new(format: TensorMapFormat, tensor: TensorArg<'a, R>, elem: Elem) -> Self {
        let TensorArg::Handle { handle, .. } = &tensor else {
            panic!("Can't use alias for TensorMap")
        };
        let rank = handle.shape.len();
        Self {
            metadata: TensorMapMeta {
                format,
                rank,
                shape: handle.shape.to_vec(),
                strides: handle.strides.to_vec(),
                elem_stride: vec![1; rank],
                interleave: TensorMapInterleave::None,
                swizzle: TensorMapSwizzle::None,
                prefetch: TensorMapPrefetch::None,
                oob_fill: OobFill::Zero,
                elem,
            },
            tensor,
        }
    }

    pub fn with_elem_stride(mut self, elem_stride: Vec<usize>) -> Self {
        self.metadata.elem_stride = elem_stride;
        self
    }

    pub fn with_interleave(mut self, interleave: TensorMapInterleave) -> Self {
        self.metadata.interleave = interleave;
        self
    }

    pub fn with_swizzle(mut self, swizzle: TensorMapSwizzle) -> Self {
        self.metadata.swizzle = swizzle;
        self
    }

    pub fn with_prefetch(mut self, prefetch: TensorMapPrefetch) -> Self {
        self.metadata.prefetch = prefetch;
        self
    }

    pub fn with_nan_fill(mut self) -> Self {
        self.metadata.oob_fill = OobFill::NaN;
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
        let tensor = builder.tensor_map(ConstantInfo::TensorMap);
        tensor.into()
    }
    fn expand_output(
        _arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E>> {
        let tensor = builder.tensor_map(ConstantInfo::TensorMap);
        tensor.into()
    }
}

impl<E: CubePrimitive> LaunchArg for TensorMap<E> {
    type RuntimeArg<'a, R: Runtime> = TensorMapArg<'a, R>;

    fn compilation_arg<R: Runtime>(_runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        TensorMapCompilationArg
    }
}

pub fn memcpy_async_bulk_commit() {
    unexpanded!()
}

pub mod memcpy_async_bulk_commit {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(TmaOps::CommitGroup)
    }
}

pub fn memcpy_async_bulk_wait(_max_pending: u32) {
    unexpanded!()
}

pub mod memcpy_async_bulk_wait {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope, max_pending: u32) {
        scope.register(TmaOps::WaitGroup { max_pending })
    }
}

pub fn memcpy_async_bulk_wait_read(_max_pending: u32) {
    unexpanded!()
}

pub mod memcpy_async_bulk_wait_read {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope, max_pending: u32) {
        scope.register(TmaOps::WaitGroupRead { max_pending })
    }
}

pub fn memcpy_async_bulk_to_global_2d<E: CubePrimitive>(
    _src: &Slice<Line<E>>,
    _dst: &mut TensorMap<E>,
    _x: i32,
    _y: i32,
) {
    unexpanded!()
}

pub mod memcpy_async_bulk_to_global_2d {
    use cubecl_ir::{Instruction, TmaOps};

    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        src: ExpandElementTyped<Slice<Line<E>>>,
        dst: ExpandElementTyped<TensorMap<E>>,
        x: ExpandElementTyped<i32>,
        y: ExpandElementTyped<i32>,
    ) {
        let source = *src.expand;
        let dst = *dst.expand;
        let coordinates = vec![*x.expand, *y.expand];
        scope.register(Instruction::new(
            TmaOps::MemCopyAsyncBulkToGlobal {
                source,
                coordinates,
            },
            dst,
        ))
    }
}

pub fn memcpy_async_bulk_to_global_3d<E: CubePrimitive>(
    _src: &Slice<Line<E>>,
    _dst: &mut TensorMap<E>,
    _x: i32,
    _y: i32,
    _z: i32,
) {
    unexpanded!()
}

pub mod memcpy_async_bulk_to_global_3d {
    use cubecl_ir::{Instruction, TmaOps};

    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        src: ExpandElementTyped<Slice<Line<E>>>,
        dst: ExpandElementTyped<TensorMap<E>>,
        x: ExpandElementTyped<i32>,
        y: ExpandElementTyped<i32>,
        z: ExpandElementTyped<i32>,
    ) {
        let source = *src.expand;
        let dst = *dst.expand;
        let coordinates = vec![*x.expand, *y.expand, *z.expand];
        scope.register(Instruction::new(
            TmaOps::MemCopyAsyncBulkToGlobal {
                source,
                coordinates,
            },
            dst,
        ))
    }
}

pub fn memcpy_async_bulk_to_global_4d<E: CubePrimitive>(
    _src: &Slice<Line<E>>,
    _dst: &mut TensorMap<E>,
    _x: i32,
    _y: i32,
    _z: i32,
    _w: i32,
) {
    unexpanded!()
}

pub mod memcpy_async_bulk_to_global_4d {
    use cubecl_ir::{Instruction, TmaOps};

    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        src: ExpandElementTyped<Slice<Line<E>>>,
        dst: ExpandElementTyped<TensorMap<E>>,
        x: ExpandElementTyped<i32>,
        y: ExpandElementTyped<i32>,
        z: ExpandElementTyped<i32>,
        w: ExpandElementTyped<i32>,
    ) {
        let source = *src.expand;
        let dst = *dst.expand;
        let coordinates = vec![*x.expand, *y.expand, *z.expand, *w.expand];
        scope.register(Instruction::new(
            TmaOps::MemCopyAsyncBulkToGlobal {
                source,
                coordinates,
            },
            dst,
        ))
    }
}

pub fn memcpy_async_bulk_to_global_5d<E: CubePrimitive>(
    _src: &Slice<Line<E>>,
    _dst: &mut TensorMap<E>,
    _x: i32,
    _y: i32,
    _z: i32,
    _w: i32,
    _v: i32,
) {
    unexpanded!()
}

pub mod memcpy_async_bulk_to_global_5d {
    use cubecl_ir::{Instruction, TmaOps};

    use super::*;

    #[allow(clippy::too_many_arguments)]
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        src: ExpandElementTyped<Slice<Line<E>>>,
        dst: ExpandElementTyped<TensorMap<E>>,
        x: ExpandElementTyped<i32>,
        y: ExpandElementTyped<i32>,
        z: ExpandElementTyped<i32>,
        w: ExpandElementTyped<i32>,
        v: ExpandElementTyped<i32>,
    ) {
        let source = *src.expand;
        let dst = *dst.expand;
        let coordinates = vec![*x.expand, *y.expand, *z.expand, *w.expand, *v.expand];
        scope.register(Instruction::new(
            TmaOps::MemCopyAsyncBulkToGlobal {
                source,
                coordinates,
            },
            dst,
        ))
    }
}
