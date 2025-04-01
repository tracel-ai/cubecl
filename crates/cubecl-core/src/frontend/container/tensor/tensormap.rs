use std::marker::PhantomData;

use crate::{ConstantInfo, ir::ExpandElement};
use crate::{prelude::*, unexpanded};
use cubecl_ir::Elem;
use cubecl_runtime::server::TensorMapMeta;
use paste::paste;
use serde::{Deserialize, Serialize};

pub use cubecl_runtime::tma::*;

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

/// A CUDA `CUtensorMap` object. Represents a tensor encoded with a lot of metadata, and is an
/// opaque packed object at runtime. Does not support retrieving any shapes or strides, nor does
/// it give access to the pointer. So these need to be passed separately in an aliased `Tensor` if needed.
///
/// Also see [cubecl_common::tma].
#[derive(Clone)]
pub struct TensorMap<E: CubePrimitive> {
    _ty: PhantomData<E>,
}

impl<E: CubePrimitive> Copy for TensorMap<E> {}

impl<E: CubePrimitive> TensorMap<E> {
    /// Create a dummy tensor map to satisfy the type checker. Not actually valid, so the code should
    /// panic if this is ever reached.
    pub fn dummy() -> Self {
        unreachable!("Dummy code")
    }

    pub fn __expand_dummy(_scope: &mut Scope) -> ExpandElementTyped<Self> {
        unreachable!("Dummy code")
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

/// Compilation argument for a [tensor map](TensorMap).
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

/// Commit an async tensor operation. Not sure how this works, poor docs. But you need to call it
/// after a write, but not after reads.
pub fn memcpy_async_tensor_commit() {
    unexpanded!()
}

pub mod memcpy_async_tensor_commit {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(TmaOps::CommitGroup)
    }
}

/// Wait until at most `max_pending` TMA copy operations are in flight.
pub fn memcpy_async_tensor_wait(_max_pending: u32) {
    unexpanded!()
}

pub mod memcpy_async_tensor_wait {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope, max_pending: u32) {
        scope.register(TmaOps::WaitGroup { max_pending })
    }
}

/// Wait TMA copy operations have finished reading from shared memory, with at most `max_pending`
/// operations being unfinished.
///
/// # Example
///
/// I believe you may use `max_pending` like this.
///
/// ```ignore
/// copy_data(smem1);
/// copy_data(smem2);
/// copy_data(smem3);
/// copy_data(smem4);
/// memcpy_async_tensor_wait_read(2);
/// // reuse smem1 & smem2 while 3 and 4 are still pending
/// ```
pub fn memcpy_async_tensor_wait_read(_max_pending: u32) {
    unexpanded!()
}

pub mod memcpy_async_tensor_wait_read {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope, max_pending: u32) {
        scope.register(TmaOps::WaitGroupRead { max_pending })
    }
}

macro_rules! copy_tensor_to_global {
    ($dim: literal, $($arg: expr),*) => {
        paste! {
            /// Copy a tile from a shared memory `src` to a global memory `dst`, with the provided
            /// offsets. Should be combined with [`memcpy_async_tensor_commit`] and
            /// [`memcpy_async_tensor_wait_read`].
            #[allow(unused)]
            pub fn [<memcpy_async_tensor_to_global_ $dim d>]<E: CubePrimitive>(
                src: &Slice<Line<E>>,
                dst: &mut TensorMap<E>,
                $($arg: i32),*
            ) {
                unexpanded!()
            }

            pub mod [<memcpy_async_tensor_to_global_ $dim d>] {
                use cubecl_ir::{Instruction, TmaOps};

                use super::*;

                #[allow(clippy::too_many_arguments)]
                pub fn expand<E: CubePrimitive>(
                    scope: &mut Scope,
                    src: ExpandElementTyped<Slice<Line<E>>>,
                    dst: ExpandElementTyped<TensorMap<E>>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    let source = *src.expand;
                    let dst = *dst.expand;
                    let coordinates = vec![$(*$arg.expand),*];
                    scope.register(Instruction::new(
                        TmaOps::MemCopyAsyncTensorToGlobal {
                            source,
                            coordinates,
                        },
                        dst,
                    ))
                }
            }
        }
    };
}

copy_tensor_to_global!(2, y, x);
copy_tensor_to_global!(3, z, y, x);
copy_tensor_to_global!(4, w, z, y, x);
copy_tensor_to_global!(5, v, w, z, y, x);
