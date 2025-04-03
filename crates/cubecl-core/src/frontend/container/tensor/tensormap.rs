use std::marker::PhantomData;

use crate::ir::ExpandElement;
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
    pub tensor: TensorArg<'a, R>,
    pub metadata: TensorMapMeta,
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
        let tensor = builder.tensor_map();
        tensor.into()
    }
    fn expand_output(
        _arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<TensorMap<E>> {
        let tensor = builder.tensor_map();
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
            /// offsets. Should be combined with [`memcpy_async_tensor_commit`] and
            /// [`memcpy_async_tensor_wait_read`].
            #[allow(unused)]
            pub fn [<tma_store_ $dim d>]<E: CubePrimitive>(
                src: &Slice<Line<E>>,
                dst: &mut TensorMap<E>,
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
                    src: ExpandElementTyped<Slice<Line<E>>>,
                    dst: ExpandElementTyped<TensorMap<E>>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    let source = *src.expand;
                    let dst = *dst.expand;
                    let coordinates = vec![$(*$arg.expand),*];
                    scope.register(Instruction::new(
                        TmaOps::TmaStore {
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

tma_store!(2, y, x);
tma_store!(3, z, y, x);
tma_store!(4, w, z, y, x);
tma_store!(5, v, w, z, y, x);

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use cubecl_ir::{ExpandElement, Item, Metadata};

    use super::*;
    use crate::{
        ir::{Arithmetic, BinaryOperator, Instruction},
        prelude::Array,
    };

    impl<T: CubePrimitive> TensorMap<T> {
        /// Obtain the stride of input at dimension dim
        pub fn stride<C: Index>(&self, _dim: C) -> u32 {
            unexpanded!()
        }

        /// Obtain the shape of input at dimension dim
        pub fn shape<C: Index>(&self, _dim: C) -> u32 {
            unexpanded!()
        }

        /// Obtain the coordinate corresponding to the given `index` of the tensor at dimension `dim`.
        ///
        /// A coordinate is a list of indices corresponding to the multi-dimensional position of an element in the tensor.
        /// The `dim` element in a coordinate is the position along the `dim` dimension of the tensor.
        pub fn coordinate<I: Index, D: Index>(&self, _index: I, _dim: D) -> u32 {
            unexpanded!()
        }

        /// The number of vectorized elements in the tensor.
        ///
        /// # Warning
        ///
        /// The length will be affected by the vectorization factor. To obtain the number of elements,
        /// you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// The length of the buffer representing the tensor in terms of vectorized elements.
        ///
        /// # Warning
        ///
        /// The buffer length will be affected by the vectorization factor. To obtain the number of
        /// elements, you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn buffer_len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> u32 {
            unexpanded!()
        }

        /// Try to cast the tensormap to the given type and panic if the type isn't the same.
        ///
        /// This function should only be used to satisfy the Rust type system, when two generic
        /// types are supposed to be the same.
        pub fn try_cast_unchecked<E: CubePrimitive>(&self) -> TensorMap<E> {
            unexpanded!()
        }

        // Expand function of [stride](Tensor::stride).
        pub fn __expand_stride<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_stride_method(scope, dim)
        }

        // Expand function of [shape](Tensor::shape).
        pub fn __expand_shape<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_shape_method(scope, dim)
        }

        // Expand function of [coordinate](Tensor::coordinate).
        pub fn __expand_coordinate<I: Index, D: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
            index: ExpandElementTyped<u32>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_coordinate_method(scope, index, dim)
        }

        // Expand function of [len](Tensor::len).
        pub fn __expand_len<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_len_method(scope)
        }

        // Expand function of [buffer_len](Tensor::buffer_len).
        pub fn __expand_buffer_len<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_buffer_len_method(scope)
        }

        // Expand function of [rank](Tensor::rank).
        pub fn __expand_rank<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_rank_method(scope)
        }
    }

    impl<T: CubePrimitive> ExpandElementTyped<TensorMap<T>> {
        // Expand method of [stride](Tensor::stride).
        pub fn __expand_stride_method(
            self,
            scope: &mut Scope,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let dim: ExpandElement = dim.into();
            let out = scope.create_local(Item::new(u32::as_elem(scope)));
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
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let dim: ExpandElement = dim.into();
            let out = scope.create_local(Item::new(u32::as_elem(scope)));
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
            index: ExpandElementTyped<u32>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let index: ExpandElement = index.into();
            let stride = self.clone().__expand_stride_method(scope, dim.clone());
            let shape = self.clone().__expand_shape_method(scope, dim.clone());

            // Compute `num_strides = index / stride`.
            let num_strides = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(
                Arithmetic::Div(BinaryOperator {
                    lhs: *index,
                    rhs: stride.expand.into(),
                }),
                num_strides.clone().into(),
            ));

            // Compute `coordinate = num_strides % shape `.
            let coordinate = scope.create_local(Item::new(u32::as_elem(scope)));
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
        pub fn __expand_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(scope)
        }

        // Expand method of [buffer_len](Tensor::buffer_len).
        pub fn __expand_buffer_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_buffer_len_method(scope)
        }

        // Expand method of [rank](Tensor::rank).
        pub fn __expand_rank_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let out = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(Metadata::Rank { var: *self.expand }, *out));
            out.into()
        }

        /// Expand method of [try_cast_unchecked](Slice::try_cast_unchecked).
        pub fn __expand_try_cast_unchecked_method<E: CubePrimitive>(
            self,
            scope: &mut Scope,
        ) -> ExpandElementTyped<TensorMap<E>> {
            if T::as_elem(scope) != E::as_elem(scope) && !is_tf32::<E, T>(scope) {
                panic!("Try cast unchecked should only be used to satisfy the rust type system.")
            }

            self.expand.into()
        }
    }
}
