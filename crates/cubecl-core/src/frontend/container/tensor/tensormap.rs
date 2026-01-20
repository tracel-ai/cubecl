use core::marker::PhantomData;

use crate as cubecl;
use crate::ir::ExpandElement;
use crate::{prelude::*, unexpanded};
use cubecl_ir::{LineSize, StorageType, Type};
use cubecl_runtime::server::TensorMapMeta;
use paste::paste;
use serde::{Deserialize, Serialize};

pub use cubecl_runtime::tma::*;

pub trait TensorMapKind: CubeType + Clone + Copy + Send + Sync + 'static {
    type Args: Clone;

    fn as_format(args: Self::Args) -> TensorMapFormat;
}

/// Regular tiled tensor map
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct Tiled {}
/// Im2col indexing. Loads a "column" (not the same column as im2col) of pixels into shared
/// memory, with a certain offset (kernel position). The corners are the bounds to load pixels
/// from *at offset 0*, so the top left corner of the kernel. The offset is added to the
/// corner offsets, so a `(-1, -1)` corner will stop the bounding box at `(1, 1)` for kernel
/// offset `(2, 2)`.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct Im2col;
/// 1D im2col, not properly supported yet
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct Im2colWide;

impl TensorMapKind for Tiled {
    type Args = TiledArgs;

    fn as_format(args: Self::Args) -> TensorMapFormat {
        TensorMapFormat::Tiled(args)
    }
}

impl TensorMapKind for Im2col {
    type Args = Im2colArgs;

    fn as_format(args: Self::Args) -> TensorMapFormat {
        TensorMapFormat::Im2col(args)
    }
}

impl TensorMapKind for Im2colWide {
    type Args = Im2colWideArgs;

    fn as_format(args: Self::Args) -> TensorMapFormat {
        TensorMapFormat::Im2colWide(args)
    }
}

/// Grid constant tensor map, currently only maps to CUDA tensormap. May be interleaved or swizzled,
/// but last dimension must be contiguous (since strides don't include the last dimension).
///
/// The tensormap is treated as an opaque type at runtime.
///
pub struct TensorMapArg<'a, R: Runtime, K: TensorMapKind> {
    pub tensor: TensorArg<'a, R>,
    pub metadata: TensorMapMeta,
    pub _kind: PhantomData<K>,
}

impl<'a, R: Runtime, K: TensorMapKind> TensorMapArg<'a, R, K> {
    pub fn new(args: K::Args, tensor: TensorArg<'a, R>, ty: StorageType) -> Self {
        let TensorArg::Handle { handle, .. } = &tensor else {
            panic!("Can't use alias for TensorMap")
        };
        let rank = handle.shape.len();
        Self {
            metadata: TensorMapMeta {
                format: K::as_format(args),
                rank,
                shape: handle.shape.to_vec(),
                strides: handle.strides.to_vec(),
                elem_stride: vec![1; rank],
                interleave: TensorMapInterleave::None,
                swizzle: TensorMapSwizzle::None,
                prefetch: TensorMapPrefetch::None,
                oob_fill: OobFill::Zero,
                storage_ty: ty,
            },
            tensor,
            _kind: PhantomData,
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
/// Also see [`cubecl_runtime::tma`].
#[derive(Clone)]
pub struct TensorMap<E: CubePrimitive, K: TensorMapKind> {
    _ty: PhantomData<E>,
    _kind: PhantomData<K>,
}

impl<E: CubePrimitive, K: TensorMapKind> Copy for TensorMap<E, K> {}

impl<E: CubePrimitive, K: TensorMapKind> TensorMap<E, K> {}

impl<E: CubePrimitive, K: TensorMapKind> ExpandElementIntoMut for TensorMap<E, K> {
    fn elem_into_mut(_scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        elem
    }
}

impl<E: CubePrimitive, K: TensorMapKind> CubeType for TensorMap<E, K> {
    type ExpandType = ExpandElementTyped<TensorMap<E, K>>;
}

impl<E: CubePrimitive, K: TensorMapKind> CubeType for *const TensorMap<E, K> {
    type ExpandType = ExpandElementTyped<TensorMap<E, K>>;
}

impl<E: CubePrimitive, K: TensorMapKind> CubeType for *mut TensorMap<E, K> {
    type ExpandType = ExpandElementTyped<TensorMap<E, K>>;
}

impl<R: Runtime, K: TensorMapKind> ArgSettings<R> for TensorMapArg<'_, R, K> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor_map(self)
    }
}

impl<E: CubePrimitive, K: TensorMapKind> Lined for TensorMap<E, K> {}
impl<E: CubePrimitive, K: TensorMapKind> LinedExpand for ExpandElementTyped<TensorMap<E, K>> {
    fn line_size(&self) -> LineSize {
        1
    }
}

/// Compilation argument for a [tensor map](TensorMap).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TensorMapCompilationArg;

impl CompilationArg for TensorMapCompilationArg {}

impl<E: CubePrimitive, K: TensorMapKind> LaunchArg for TensorMap<E, K> {
    type RuntimeArg<'a, R: Runtime> = TensorMapArg<'a, R, K>;
    type CompilationArg = TensorMapCompilationArg;

    fn compilation_arg<R: Runtime>(_runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        TensorMapCompilationArg
    }

    fn expand(
        _arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E, K>> {
        let tensor = builder.input_tensor_map(Type::new(E::as_type(&builder.scope)));
        tensor.into()
    }
    fn expand_output(
        _arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E, K>> {
        let tensor = builder.output_tensor_map(Type::new(E::as_type(&builder.scope)));
        tensor.into()
    }
}

/// Commit an async tensor operation. Not sure how this works, poor docs. But you need to call it
/// after a write, but not after reads.
pub fn tma_group_commit() {
    unexpanded!()
}

pub mod tma_group_commit {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(TmaOps::CommitGroup)
    }
}

/// Wait until at most `max_pending` TMA copy operations are in flight.
pub fn tma_group_wait(_max_pending: u32) {
    unexpanded!()
}

pub mod tma_group_wait {
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
/// tma_wait_read(2);
/// // reuse smem1 & smem2 while 3 and 4 are still pending
/// ```
pub fn tma_group_wait_read(_max_pending: u32) {
    unexpanded!()
}

pub mod tma_group_wait_read {
    use cubecl_ir::TmaOps;

    use super::*;

    pub fn expand(scope: &mut Scope, max_pending: u32) {
        scope.register(TmaOps::WaitGroupRead { max_pending })
    }
}

macro_rules! tma_store {
    ($dim: literal, $($arg: expr),*) => {
        paste! {
            /// Copy a tile from a shared memory `src` to a global memory `dst`, with the provided
            /// offsets. Should be combined with ``memcpy_async_tensor_commit`` and
            /// ``memcpy_async_tensor_wait_read``.
            #[allow(unused)]
            pub fn [<tma_store_ $dim d>]<E: CubePrimitive>(
                src: &Slice<Line<E>>,
                dst: &mut TensorMap<E, Tiled>,
                $($arg: i32),*
            ) {
                unexpanded!()
            }

            pub mod [<tma_store_ $dim d>] {
                use cubecl_ir::{Instruction, TmaOps};

                use super::*;

                #[allow(clippy::too_many_arguments)]
                pub fn expand<E: CubePrimitive>(
                    scope: &mut Scope,
                    src: SliceExpand<Line<E>, ReadOnly>,
                    dst: ExpandElementTyped<TensorMap<E, Tiled>>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    let (source, source_offset) = src.__to_raw_parts();
                    let dst = *dst.expand;
                    let coordinates = vec![$(*$arg.expand),*];
                    scope.register(Instruction::new(
                        TmaOps::TmaStore {
                            source,
                            coordinates,
                            offset_source: source_offset,
                        },
                        dst,
                    ))
                }
            }
        }
    };
}

tma_store!(1, x);
tma_store!(2, y, x);
tma_store!(3, z, y, x);
tma_store!(4, w, z, y, x);
tma_store!(5, v, w, z, y, x);

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use cubecl_ir::{ExpandElement, Metadata, Type, VariableKind};

    use super::*;
    use crate::{
        ir::{Arithmetic, BinaryOperator, Instruction},
        prelude::Array,
    };

    impl<T: CubePrimitive, K: TensorMapKind> TensorMap<T, K> {
        /// Get a reference to the underlying buffer for the tensor map.
        pub fn buffer(&self) -> Tensor<Line<T>> {
            unexpanded!()
        }

        /// Obtain the stride of input at dimension dim
        pub fn stride(&self, _dim: usize) -> usize {
            unexpanded!()
        }

        /// Obtain the shape of input at dimension dim
        pub fn shape(&self, _dim: usize) -> usize {
            unexpanded!()
        }

        /// Obtain the coordinate corresponding to the given `index` of the tensor at dimension `dim`.
        ///
        /// A coordinate is a list of indices corresponding to the multi-dimensional position of an element in the tensor.
        /// The `dim` element in a coordinate is the position along the `dim` dimension of the tensor.
        pub fn coordinate(&self, _index: usize, _dim: usize) -> usize {
            unexpanded!()
        }

        /// The number of vectorized elements in the tensor.
        ///
        /// # Warning
        ///
        /// The length will be affected by the vectorization factor. To obtain the number of elements,
        /// you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> usize {
            unexpanded!()
        }

        /// The length of the buffer representing the tensor in terms of vectorized elements.
        ///
        /// # Warning
        ///
        /// The buffer length will be affected by the vectorization factor. To obtain the number of
        /// elements, you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn buffer_len(&self) -> usize {
            unexpanded!()
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> usize {
            unexpanded!()
        }

        /// Downcast the tensormap to the given type and panic if the type isn't the same.
        ///
        /// This function should only be used to satisfy the Rust type system, when two generic
        /// types are supposed to be the same.
        pub fn downcast<E: CubePrimitive>(&self) -> TensorMap<E, K> {
            unexpanded!()
        }

        // Expand function of [buffer](TensorMap::buffer).
        pub fn __expand_buffer(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
        ) -> ExpandElementTyped<Tensor<Line<T>>> {
            expand.__expand_buffer_method(scope)
        }

        // Expand function of [stride](TensorMap::stride).
        pub fn __expand_stride(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
            dim: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            expand.__expand_stride_method(scope, dim)
        }

        // Expand function of [shape](TensorMap::shape).
        pub fn __expand_shape(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
            dim: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            expand.__expand_shape_method(scope, dim)
        }

        // Expand function of [coordinate](TensorMap::coordinate).
        pub fn __expand_coordinate(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
            index: ExpandElementTyped<usize>,
            dim: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            expand.__expand_coordinate_method(scope, index, dim)
        }

        // Expand function of [len](TensorMap::len).
        pub fn __expand_len(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
        ) -> ExpandElementTyped<usize> {
            expand.__expand_len_method(scope)
        }

        // Expand function of [buffer_len](TensorMap::buffer_len).
        pub fn __expand_buffer_len(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
        ) -> ExpandElementTyped<usize> {
            expand.__expand_buffer_len_method(scope)
        }

        // Expand function of [rank](TensorMap::rank).
        pub fn __expand_rank(
            scope: &mut Scope,
            expand: ExpandElementTyped<TensorMap<T, K>>,
        ) -> ExpandElementTyped<usize> {
            expand.__expand_rank_method(scope)
        }
    }

    impl<T: CubePrimitive, K: TensorMapKind> ExpandElementTyped<TensorMap<T, K>> {
        // Expand method of [buffer](TensorMap::buffer).
        pub fn __expand_buffer_method(
            self,
            scope: &mut Scope,
        ) -> ExpandElementTyped<Tensor<Line<T>>> {
            let tensor = match self.expand.kind {
                VariableKind::TensorMapInput(id) => scope.input(id, self.expand.ty),
                VariableKind::TensorMapOutput(id) => scope.output(id, self.expand.ty),
                _ => unreachable!(),
            };
            tensor.into()
        }

        // Expand method of [stride](Tensor::stride).
        pub fn __expand_stride_method(
            self,
            scope: &mut Scope,
            dim: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            let dim: ExpandElement = dim.into();
            let out = scope.create_local(Type::new(usize::as_type(scope)));
            scope.register(Instruction::new(
                Metadata::Stride {
                    dim: *dim,
                    var: self.expand.into(),
                },
                out.clone().into(),
            ));
            out.into()
        }

        // Expand method of [shape](Tensor::shape).
        pub fn __expand_shape_method(
            self,
            scope: &mut Scope,
            dim: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            let dim: ExpandElement = dim.into();
            let out = scope.create_local(Type::new(usize::as_type(scope)));
            scope.register(Instruction::new(
                Metadata::Shape {
                    dim: *dim,
                    var: self.expand.into(),
                },
                out.clone().into(),
            ));
            out.into()
        }

        // Expand method of [coordinate](Tensor::coordinate).
        pub fn __expand_coordinate_method(
            self,
            scope: &mut Scope,
            index: ExpandElementTyped<usize>,
            dim: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            let index: ExpandElement = index.into();
            let stride = self.clone().__expand_stride_method(scope, dim.clone());
            let shape = self.clone().__expand_shape_method(scope, dim.clone());

            // Compute `num_strides = index / stride`.
            let num_strides = scope.create_local(Type::new(usize::as_type(scope)));
            scope.register(Instruction::new(
                Arithmetic::Div(BinaryOperator {
                    lhs: *index,
                    rhs: stride.expand.into(),
                }),
                num_strides.clone().into(),
            ));

            // Compute `coordinate = num_strides % shape `.
            let coordinate = scope.create_local(Type::new(usize::as_type(scope)));
            scope.register(Instruction::new(
                Arithmetic::Modulo(BinaryOperator {
                    lhs: *num_strides,
                    rhs: shape.expand.into(),
                }),
                coordinate.clone().into(),
            ));

            coordinate.into()
        }

        // Expand method of [len](Tensor::len).
        pub fn __expand_len_method(self, scope: &mut Scope) -> ExpandElementTyped<usize> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(scope)
        }

        // Expand method of [buffer_len](Tensor::buffer_len).
        pub fn __expand_buffer_len_method(self, scope: &mut Scope) -> ExpandElementTyped<usize> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_buffer_len_method(scope)
        }

        // Expand method of [rank](Tensor::rank).
        pub fn __expand_rank_method(self, scope: &mut Scope) -> ExpandElementTyped<usize> {
            let out = scope.create_local(Type::new(u32::as_type(scope)));
            scope.register(Instruction::new(Metadata::Rank { var: *self.expand }, *out));
            out.into()
        }

        /// Expand method of [`TensorMap::downcast`].
        pub fn __expand_downcast_method<E: CubePrimitive>(
            self,
            scope: &mut Scope,
        ) -> ExpandElementTyped<TensorMap<E, K>> {
            if T::as_type(scope) != E::as_type(scope) && !is_tf32::<E, T>(scope) {
                panic!("Downcast should only be used to satisfy the Rust type system.")
            }

            self.expand.into()
        }
    }
}
