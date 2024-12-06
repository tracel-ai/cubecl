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
    ir::{self, Instruction, Operation},
    unexpanded,
};

use super::{
    CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementBaseInit, ExpandElementTyped,
    IntoRuntime, Slice, SliceMut,
};

pub use ir::{MatrixIdent, MatrixLayout};

/// A matrix represent a 2D grid of numbers.
///
/// They can either be in a [row major](MatrixLayout::RowMajor) or a
/// [column major](MatrixLayout::ColMajor) format.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Matrix<C: CubePrimitive> {
    _c: PhantomData<C>,
}

impl<C: CubePrimitive> CubeType for Matrix<C> {
    type ExpandType = ExpandElementTyped<Matrix<C>>;
}

impl<C: CubePrimitive> ExpandElementBaseInit for Matrix<C> {
    fn init_elem(_context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        elem.into()
    }
}

impl<C: CubePrimitive> IntoRuntime for Matrix<C> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        unimplemented!("Matrices can't exist at compile time")
    }
}

// impl<C: CubePrimitive> Init for ExpandElementTyped<Matrix<C>> {
//     fn init(self, _context: &mut CubeContext) -> Self {
//         self
//     }
// }

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
        value: &Slice<C>,
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
    ) -> ExpandElementTyped<Matrix<C>> {
        let elem = context.create_matrix(ir::Matrix {
            ident,
            m: m.constant().unwrap().as_u32() as u8,
            n: n.constant().unwrap().as_u32() as u8,
            k: k.constant().unwrap().as_u32() as u8,
            elem: C::as_elem(),
            layout,
        });

        elem.into()
    }

    pub fn __expand_from_value(
        context: &mut CubeContext,
        ident: MatrixIdent,
        m: ExpandElementTyped<u32>,
        n: ExpandElementTyped<u32>,
        k: ExpandElementTyped<u32>,
        layout: MatrixLayout,
        value: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<Matrix<C>> {
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
        value: ExpandElementTyped<Slice<C>>,
        stride: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Matrix<C>> {
        let mat = Self::__expand_uninitialized(context, ident, m, n, k, layout);
        load::expand(context, mat.clone(), value, stride);
        mat
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: CubePrimitive>(mat: &Matrix<C>, value: C) {
    unexpanded!()
}

/// Module containing the expand function for [fill()].
pub mod fill {
    use super::*;

    /// Expand method of [fill()].
    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        mat: ExpandElementTyped<Matrix<C>>,
        value: ExpandElementTyped<C>,
    ) {
        let value: ExpandElement = value.into();
        let mat: ExpandElement = mat.into();

        context.register(Instruction::new(ir::CoopMma::Fill { value: *value }, *mat));
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
        context: &mut CubeContext,
        mat: ExpandElementTyped<Matrix<C>>,
        value: ExpandElementTyped<Slice<V>>,
        stride: ExpandElementTyped<u32>,
    ) {
        let stride: ExpandElement = stride.into();
        let mat: ExpandElement = mat.into();
        let out = *mat;

        let ident = match out.kind {
            ir::VariableKind::Matrix { id, mat, depth } => mat.ident,
            _ => unreachable!("{:?}", out.kind),
        };
        assert_ne!(
            ident,
            MatrixIdent::Accumulator,
            "Loading accumulator requires explicit layout. Use `load_with_layout` instead."
        );

        context.register(Instruction::new(
            ir::CoopMma::Load {
                value: *value.expand,
                stride: *stride,
                layout: None,
            },
            out,
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
    pub fn expand<C: CubePrimitive, V: CubePrimitive>(
        context: &mut CubeContext,
        mat: ExpandElementTyped<Matrix<C>>,
        value: ExpandElementTyped<Slice<V>>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();
        let mat: ExpandElement = mat.into();

        context.register(Instruction::new(
            ir::CoopMma::Load {
                value: *value.expand,
                stride: *stride,
                layout: Some(layout),
            },
            *mat,
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
    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive>(
        context: &mut CubeContext,
        output: ExpandElementTyped<SliceMut<O>>,
        mat: ExpandElementTyped<Matrix<C>>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();
        let mat: ExpandElement = mat.into();

        context.register(Instruction::new(
            ir::CoopMma::Store {
                mat: *mat,
                stride: *stride,
                layout,
            },
            *output.expand,
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
        context: &mut CubeContext,
        mat_a: ExpandElementTyped<Matrix<A>>,
        mat_b: ExpandElementTyped<Matrix<B>>,
        mat_c: ExpandElementTyped<Matrix<C>>,
        mat_d: ExpandElementTyped<Matrix<D>>,
    ) {
        let mat_a: ExpandElement = mat_a.into();
        let mat_b: ExpandElement = mat_b.into();
        let mat_c: ExpandElement = mat_c.into();
        let mat_d: ExpandElement = mat_d.into();

        context.register(Instruction::new(
            ir::CoopMma::Execute {
                mat_a: *mat_a,
                mat_b: *mat_b,
                mat_c: *mat_c,
            },
            *mat_d,
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
        context: &mut CubeContext,
        input: ExpandElementTyped<Matrix<C>>,
    ) -> ExpandElementTyped<Matrix<O>> {
        let input: ExpandElement = input.into();

        let input_mat = match input.kind {
            ir::VariableKind::Matrix { id, mat, depth } => mat,
            _ => unreachable!(),
        };

        if core::any::TypeId::of::<C>() == core::any::TypeId::of::<O>() {
            return input.into();
        }

        let output = context.create_matrix(ir::Matrix {
            ident: input_mat.ident,
            m: input_mat.m,
            n: input_mat.n,
            k: input_mat.k,
            elem: O::as_elem(),
            layout: MatrixLayout::Undefined,
        });

        context.register(Instruction::new(
            ir::CoopMma::Cast { input: *input },
            *output,
        ));

        output.into()
    }
}

impl From<ir::CoopMma> for Operation {
    fn from(value: ir::CoopMma) -> Self {
        Operation::CoopMma(value)
    }
}
