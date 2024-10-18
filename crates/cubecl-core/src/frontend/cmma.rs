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

use std::marker::PhantomData;

use crate::{
    ir::{self, Operation},
    unexpanded,
};

use super::{
    CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementTyped, Init, IntoRuntime,
    Slice, SliceMut,
};

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
#[derive(Clone)]
pub struct MatrixExpand {
    elem: ExpandElement,
    ident: MatrixIdent,
}

impl<C: CubeType> CubeType for Matrix<C> {
    type ExpandType = MatrixExpand;
}

impl<C: CubeType> IntoRuntime for Matrix<C> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> MatrixExpand {
        unimplemented!("Matrices can't exist at compile time")
    }
}

impl Init for MatrixExpand {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

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
        ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
    ) -> Self {
        Matrix { _c: PhantomData }
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
        ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
        value: C,
    ) -> Self {
        Matrix { _c: PhantomData }
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
        ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
        value: &Slice<'_, C>,
        stride: u32,
    ) -> Self {
        Matrix { _c: PhantomData }
    }

    pub fn __expand_uninitialized(
        context: &mut CubeContext,
        ident: MatrixIdent,
        m: ExpandElementTyped<u32>,
        n: ExpandElementTyped<u32>,
        k: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) -> MatrixExpand {
        let elem = context.create_matrix(ir::Matrix {
            ident,
            m: m.constant().unwrap().as_u32() as u8,
            n: n.constant().unwrap().as_u32() as u8,
            k: k.constant().unwrap().as_u32() as u8,
            elem: C::as_elem(),
            layout,
        });
        MatrixExpand { elem, ident }
    }

    pub fn __expand_from_value(
        context: &mut CubeContext,
        ident: MatrixIdent,
        m: ExpandElementTyped<u32>,
        n: ExpandElementTyped<u32>,
        k: ExpandElementTyped<u32>,
        layout: MatrixLayout,
        value: ExpandElementTyped<C>,
    ) -> MatrixExpand {
        let mat = Self::__expand_uninitialized(context, ident, m, n, k, layout);
        fill::expand(context, mat.clone(), value);
        mat
    }

    #[allow(clippy::too_many_arguments)]
    pub fn __expand_from_slice(
        context: &mut CubeContext,
        ident: MatrixIdent,
        m: ExpandElementTyped<u32>,
        n: ExpandElementTyped<u32>,
        k: ExpandElementTyped<u32>,
        layout: MatrixLayout,
        value: ExpandElementTyped<Slice<'static, C>>,
        stride: ExpandElementTyped<u32>,
    ) -> MatrixExpand {
        let mat = Self::__expand_uninitialized(context, ident, m, n, k, layout);
        load::expand(context, mat.clone(), value, stride);
        mat
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
        context: &mut CubeContext,
        mat: MatrixExpand,
        value: ExpandElementTyped<C>,
    ) {
        let value: ExpandElement = value.into();
        context.register(Operation::CoopMma(ir::CoopMma::Fill {
            mat: *mat.elem,
            value: *value,
        }));
    }
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: CubeType>(mat: &Matrix<C>, value: &Slice<'_, C>, stride: u32) {
    unexpanded!()
}

/// Module containing the expand function for [load()].
pub mod load {
    use super::*;

    /// Expand method of [load()].
    #[allow(unused_variables)]
    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        mat: MatrixExpand,
        value: ExpandElementTyped<Slice<'static, C>>,
        stride: ExpandElementTyped<u32>,
    ) {
        let stride: ExpandElement = stride.into();
        assert_ne!(
            mat.ident,
            MatrixIdent::Accumulator,
            "Loading accumulator requires explicit layout. Use `load_with_layout` instead."
        );

        context.register(Operation::CoopMma(ir::CoopMma::Load {
            mat: *mat.elem,
            value: *value.expand,
            stride: *stride,
            layout: None,
        }));
    }
}

/// Load the matrix with the provided array using the stride with an explicit layout.
/// Explicit layouts are required when loading accumulators.
#[allow(unused_variables)]
pub fn load_with_layout<C: CubeType>(
    mat: &Matrix<C>,
    value: &Slice<'_, C>,
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
    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        mat: MatrixExpand,
        value: ExpandElementTyped<Slice<'static, C>>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();

        context.register(Operation::CoopMma(ir::CoopMma::Load {
            mat: *mat.elem,
            value: *value.expand,
            stride: *stride,
            layout: Some(layout),
        }));
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn store<C: CubePrimitive>(
    output: &mut SliceMut<'_, C>,
    mat: &Matrix<C>,
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [store()].
pub mod store {
    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        output: ExpandElementTyped<SliceMut<'static, C>>,
        mat: MatrixExpand,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();

        context.register(Operation::CoopMma(ir::CoopMma::Store {
            output: *output.expand,
            mat: *mat.elem,
            stride: *stride,
            layout,
        }));
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
        context: &mut CubeContext,
        mat_a: MatrixExpand,
        mat_b: MatrixExpand,
        mat_c: MatrixExpand,
        mat_d: MatrixExpand,
    ) {
        context.register(Operation::CoopMma(ir::CoopMma::Execute {
            mat_a: *mat_a.elem,
            mat_b: *mat_b.elem,
            mat_c: *mat_c.elem,
            mat_d: *mat_d.elem,
        }));
    }
}
