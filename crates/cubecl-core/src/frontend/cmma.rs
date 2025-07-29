//! This module exposes cooperative matrix-multiply and accumulate operations.
//!
//! Most of the functions are actually unsafe, since they mutate their input, even if they are
//! passed as reference.
//!
//! # Example
//!
//! This is a basic 16x16x16 matrix multiplication example.
//!
//! ```rust, ignore
//! #[cube(launch)]
//! pub fn example(lhs: &Array<F16>, rhs: &Array<F16>, out: &mut Array<F32>) {
//!     let a = cmma::Matrix::<F16>::new(
//!         cmma::MatrixIdent::A,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::RowMajor,
//!     );
//!     let b = cmma::Matrix::<F16>::new(
//!         cmma::MatrixIdent::B,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::ColMajor,
//!     );
//!     let c = cmma::Matrix::<F32>::new(
//!         cmma::MatrixIdent::Accumulator,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::Undefined,
//!     );
//!     cmma::fill::<F32>(&c, F32::new(0.0));
//!     cmma::load::<F16>(&a, lhs.as_slice(), u32::new(16));
//!     cmma::load::<F16>(&b, rhs.as_slice(), u32::new(16));
//!
//!     cmma::execute::<F16, F16, F32, F32>(&a, &b, &c, &c);
//!
//!     cmma::store::<F32>(
//!         out.as_slice_mut(),
//!         &c,
//!         u32::new(16),
//!         cmma::MatrixLayout::RowMajor,
//!     );
//! }
//! ```

use super::{
    CubeDebug, CubePrimitive, CubeType, ExpandElementTyped, IntoMut, ReadOnly, Slice, SliceExpand,
    SliceMut,
};
use crate as cubecl;
use crate::{
    ir::{self, Instruction},
    unexpanded,
};
use cubecl_macros::{cube, intrinsic};
use std::marker::PhantomData;

use cubecl_ir::{ExpandElement, Scope};
pub use ir::{MatrixIdent, MatrixLayout};

/// A matrix represent a 2D grid of numbers.
///
/// They can either be in a [row major](MatrixLayout::RowMajor) or a
/// [column major](MatrixLayout::ColMajor) format.
#[derive(Copy, Clone)]
pub struct Matrix<C: CubeType> {
    _c: PhantomData<C>,
}

/// Expand type of [Matrix].
pub struct MatrixExpand<C: CubeType> {
    elem: ExpandElement,
    ident: MatrixIdent,
    _c: PhantomData<C>,
}

impl<C: CubeType> Clone for MatrixExpand<C> {
    fn clone(&self) -> Self {
        Self {
            elem: self.elem.clone(),
            ident: self.ident,
            _c: self._c,
        }
    }
}

impl<C: CubeType> CubeType for Matrix<C> {
    type ExpandType = MatrixExpand<C>;
}

impl<C: CubeType> IntoMut for MatrixExpand<C> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<C: CubeType> CubeDebug for MatrixExpand<C> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.elem, name);
    }
}

#[cube]
impl<C: CubePrimitive> Matrix<C> {
    /// Create a new uninitialized matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function.
    ///
    /// # Safety
    /// Must be initialized with `load` or `fill` before use. Using it without initialization is
    /// undefined behaviour on CUDA, and completely invalid on Vulkan.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub unsafe fn uninitialized(
        #[comptime] ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
    ) -> Self {
        intrinsic!(|scope| {
            let elem = C::as_elem(scope);
            let elem = scope.create_matrix(ir::Matrix {
                ident,
                m: m.constant().unwrap().as_u32() as u8,
                n: n.constant().unwrap().as_u32() as u8,
                k: k.constant().unwrap().as_u32() as u8,
                elem,
                layout,
            });
            MatrixExpand {
                elem,
                ident,
                _c: PhantomData,
            }
        })
    }

    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function and is filled with `value`.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn from_value(
        #[comptime] ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
        value: C,
    ) -> Self {
        let mat = unsafe { Self::uninitialized(ident, m, n, k, layout) };

        intrinsic!(|scope| {
            fill::expand(scope, mat.clone(), value);
            mat
        })
    }

    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function and is loaded from `value` with `stride`.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn from_slice(
        #[comptime] ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
        value: &Slice<C>,
        stride: u32,
    ) -> Self {
        let mat = unsafe { Self::uninitialized(ident, m, n, k, layout) };

        intrinsic!(|scope| {
            load::expand(scope, mat.clone(), value, stride);
            mat
        })
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: CubeType>(mat: &Matrix<C>, value: C) {
    unexpanded!()
}

/// Module containing the expand function for [fill()].
pub mod fill {
    use super::*;

    /// Expand method of [fill()].
    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        mat: MatrixExpand<C>,
        value: ExpandElementTyped<C>,
    ) {
        let value: ExpandElement = value.into();
        scope.register(Instruction::new(
            ir::CoopMma::Fill { value: *value },
            *mat.elem,
        ));
    }
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: CubePrimitive, V: CubePrimitive>(mat: &Matrix<C>, value: &Slice<V>, stride: u32) {
    unexpanded!()
}

/// Module containing the expand function for [load()].
pub mod load {
    use super::*;

    /// Expand method of [load()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, V: CubePrimitive>(
        scope: &mut Scope,
        mat: MatrixExpand<C>,
        value: SliceExpand<V, ReadOnly>,
        stride: ExpandElementTyped<u32>,
    ) {
        let stride: ExpandElement = stride.into();
        assert_ne!(
            mat.ident,
            MatrixIdent::Accumulator,
            "Loading accumulator requires explicit layout. Use `load_with_layout` instead."
        );

        let (value, offset) = value.__to_raw_parts();

        scope.register(Instruction::new(
            ir::CoopMma::Load {
                value,
                stride: *stride,
                offset,
                layout: None,
            },
            *mat.elem,
        ));
    }
}

/// Load the matrix with the provided array using the stride with an explicit layout.
/// Explicit layouts are required when loading accumulators.
#[allow(unused_variables)]
pub fn load_with_layout<C: CubePrimitive, V: CubePrimitive>(
    mat: &Matrix<C>,
    value: &Slice<V>,
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [load_with_layout()].
pub mod load_with_layout {
    use super::*;

    /// Expand method of [load_with_layout()].
    #[allow(unused_variables)]
    pub fn expand<C: CubeType, V: CubePrimitive>(
        scope: &mut Scope,
        mat: MatrixExpand<C>,
        value: SliceExpand<V, ReadOnly>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();
        let (value, offset) = value.__to_raw_parts();

        scope.register(Instruction::new(
            ir::CoopMma::Load {
                value,
                stride: *stride,
                offset,
                layout: Some(layout),
            },
            *mat.elem,
        ));
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn store<C: CubePrimitive, O: CubePrimitive>(
    output: &mut SliceMut<O>,
    mat: &Matrix<C>,
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [store()].
pub mod store {
    use crate::prelude::ReadWrite;

    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive>(
        scope: &mut Scope,
        output: SliceExpand<O, ReadWrite>,
        mat: MatrixExpand<C>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();

        let (output, offset) = output.__to_raw_parts();

        scope.register(Instruction::new(
            ir::CoopMma::Store {
                mat: *mat.elem,
                offset,
                stride: *stride,
                layout,
            },
            output,
        ));
    }
}

/// Execute the matrix-multiply and accumulate operation on the given [matrices](Matrix).
#[allow(unused_variables)]
pub fn execute<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
    mat_a: &Matrix<A>,
    mat_b: &Matrix<B>,
    mat_c: &Matrix<C>,
    mat_d: &Matrix<D>,
) {
    unexpanded!()
}

/// Module containing the expand function for [execute()].
pub mod execute {
    use super::*;

    /// Expand method of [execute()].
    pub fn expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
        scope: &mut Scope,
        mat_a: MatrixExpand<A>,
        mat_b: MatrixExpand<B>,
        mat_c: MatrixExpand<C>,
        mat_d: MatrixExpand<D>,
    ) {
        scope.register(Instruction::new(
            ir::CoopMma::Execute {
                mat_a: *mat_a.elem,
                mat_b: *mat_b.elem,
                mat_c: *mat_c.elem,
            },
            *mat_d.elem,
        ));
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn cast<C: CubePrimitive, O: CubePrimitive>(input: &Matrix<C>) -> Matrix<O> {
    unexpanded!()
}

/// Module containing the expand function for [store()].
pub mod cast {
    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive>(
        scope: &mut Scope,
        input: MatrixExpand<C>,
    ) -> MatrixExpand<O> {
        let ident = input.ident;

        if core::any::TypeId::of::<C>() == core::any::TypeId::of::<O>() {
            return MatrixExpand {
                elem: input.elem,
                ident,
                _c: PhantomData,
            };
        }
        let input = *input.elem;
        let input_mat = match input.kind {
            ir::VariableKind::Matrix { mat, .. } => mat,
            _ => unreachable!(),
        };

        let elem = O::as_elem(scope);
        let elem = scope.create_matrix(ir::Matrix {
            ident,
            m: input_mat.m,
            n: input_mat.n,
            k: input_mat.k,
            elem,
            layout: MatrixLayout::Undefined,
        });

        let output = MatrixExpand {
            ident,
            elem,
            _c: PhantomData,
        };
        scope.register(Instruction::new(ir::CoopMma::Cast { input }, *output.elem));

        output
    }
}

impl CubeType for MatrixLayout {
    type ExpandType = Self;
}

impl IntoMut for MatrixLayout {
    fn into_mut(self, _scope: &mut crate::ir::Scope) -> Self {
        self
    }
}

impl CubeDebug for MatrixLayout {}
