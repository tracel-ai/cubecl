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
//!     cmma::load::<F16>(&a, lhs.as_slice(), UInt::new(16));
//!     cmma::load::<F16>(&b, rhs.as_slice(), UInt::new(16));
//!
//!     cmma::execute::<F16, F16, F32, F32>(&a, &b, &c, &c);
//!
//!     cmma::store::<F32>(
//!         out.as_slice_mut(),
//!         &c,
//!         UInt::new(16),
//!         cmma::MatrixLayout::RowMajor,
//!     );
//! }
//! ```

use std::{marker::PhantomData, num::NonZero};

use crate::{
    ir::{self, Elem, Operation},
    new_ir::{
        Container, Expr, Expression, SquareType, StaticExpand, StaticExpanded, Strided,
        Vectorization,
    },
    prelude::*,
    unexpanded,
};

pub use ir::{MatrixIdent, MatrixLayout};

/// A matrix represent a 2D grid of numbers.
///
/// They can either be in a [row major](MatrixLayout::RowMajor) or a
/// [column major](MatrixLayout::ColMajor) format.
#[derive(Copy, Clone)]
pub struct Matrix<C: SquareType> {
    pub ident: MatrixIdent,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub layout: MatrixLayout,
    _c: PhantomData<C>,
}

impl<C: SquareType> StaticExpand for Matrix<C> {
    type Expanded = Self;
}
impl<C: SquareType> StaticExpanded for Matrix<C> {
    type Unexpanded = Self;
}
impl<C: SquareType> SquareType for Matrix<C> {
    fn ir_type() -> Elem {
        C::ir_type()
    }
}

impl<C: SquareType> Matrix<C> {
    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function.
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
    pub fn new(ident: MatrixIdent, m: u8, n: u8, k: u8, layout: MatrixLayout) -> Self {
        Self {
            ident,
            m,
            n,
            k,
            layout,
            _c: PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CmmaExpression {
    Init {
        ident: MatrixIdent,
        m: u8,
        n: u8,
        k: u8,
        layout: MatrixLayout,
        ty: Elem,
    },
    Fill {
        matrix: Box<Expression>,
        value: Box<Expression>,
    },
    Load {
        matrix: Box<Expression>,
        values: Box<Expression>,
        stride: Box<Expression>,
    },
    Store {
        matrix: Box<Expression>,
        out: Box<Expression>,
        stride: Box<Expression>,
        layout: MatrixLayout,
    },
    Execute {
        mat_a: Box<Expression>,
        mat_b: Box<Expression>,
        mat_c: Box<Expression>,
        mat_d: Box<Expression>,
    },
}

impl CmmaExpression {
    pub fn ir_type(&self) -> Elem {
        match self {
            CmmaExpression::Init { ty, .. } => *ty,
            CmmaExpression::Fill { value, .. } => value.ir_type(),
            CmmaExpression::Load { matrix, .. } => matrix.ir_type(),
            CmmaExpression::Store { matrix, .. } => matrix.ir_type(),
            CmmaExpression::Execute { .. } => Elem::Unit,
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        None
    }

    pub fn deep_clone(&self) -> Self {
        match self {
            CmmaExpression::Init { .. } => self.clone(),
            CmmaExpression::Fill { matrix, value } => CmmaExpression::Fill {
                matrix: Box::new(matrix.deep_clone()),
                value: Box::new(value.deep_clone()),
            },
            CmmaExpression::Load {
                matrix,
                values,
                stride,
            } => CmmaExpression::Load {
                matrix: Box::new(matrix.deep_clone()),
                values: Box::new(values.deep_clone()),
                stride: Box::new(stride.deep_clone()),
            },
            CmmaExpression::Store {
                matrix,
                out,
                stride,
                layout,
            } => CmmaExpression::Store {
                matrix: Box::new(matrix.deep_clone()),
                out: Box::new(out.deep_clone()),
                stride: Box::new(stride.deep_clone()),
                layout: *layout,
            },
            CmmaExpression::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => CmmaExpression::Execute {
                mat_a: Box::new(mat_a.deep_clone()),
                mat_b: Box::new(mat_b.deep_clone()),
                mat_c: Box::new(mat_c.deep_clone()),
                mat_d: Box::new(mat_d.deep_clone()),
            },
        }
    }

    pub fn flatten(self, context: &mut CubeContext) -> Option<ExpandElement> {
        match self {
            CmmaExpression::Init {
                ident,
                m,
                n,
                k,
                layout,
                ty,
            } => context
                .create_matrix(ir::Matrix {
                    ident,
                    m,
                    n,
                    k,
                    elem: ty,
                    layout,
                })
                .into(),
            CmmaExpression::Fill { matrix, value } => {
                let value = value.flatten(context).unwrap().into_variable();
                let matrix = matrix.flatten(context).unwrap().as_variable();
                context.register(Operation::CoopMma(ir::CoopMma::Fill { mat: matrix, value }));
                None
            }
            CmmaExpression::Load {
                matrix,
                values,
                stride,
            } => {
                let stride = stride.flatten(context).unwrap().into_variable();
                let value = values.flatten(context).unwrap().as_variable();
                let mat = matrix.flatten(context).unwrap().as_variable();
                context.register(Operation::CoopMma(ir::CoopMma::Load { mat, value, stride }));
                None
            }
            CmmaExpression::Store {
                matrix,
                out,
                stride,
                layout,
            } => {
                let stride = stride.flatten(context).unwrap().into_variable();
                let output = out.flatten(context).unwrap().as_variable();
                let mat = matrix.flatten(context).unwrap().as_variable();
                context.register(Operation::CoopMma(ir::CoopMma::Store {
                    mat,
                    output,
                    stride,
                    layout,
                }));
                None
            }
            CmmaExpression::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => {
                let mat_a = mat_a.flatten(context).unwrap().as_variable();
                let mat_b = mat_b.flatten(context).unwrap().as_variable();
                let mat_c = mat_c.flatten(context).unwrap().as_variable();
                let mat_d = mat_d.flatten(context).unwrap().as_variable();
                context.register(Operation::CoopMma(ir::CoopMma::Execute {
                    mat_a,
                    mat_b,
                    mat_c,
                    mat_d,
                }));
                None
            }
        }
    }
}

impl<T: SquareType> Expr for Matrix<T> {
    type Output = Matrix<T>;

    fn expression_untyped(&self) -> Expression {
        CmmaExpression::Init {
            ident: self.ident,
            m: self.m,
            n: self.n,
            k: self.k,
            layout: self.layout,
            ty: T::ir_type(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl<T: SquareType> Expr for &Matrix<T> {
    type Output = Matrix<T>;

    fn expression_untyped(&self) -> Expression {
        Matrix::<T>::expression_untyped(self)
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl<T: SquareType> Expr for &mut Matrix<T> {
    type Output = Matrix<T>;

    fn expression_untyped(&self) -> Expression {
        Matrix::<T>::expression_untyped(self)
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: SquareType>(mat: &Matrix<C>, value: C) {
    unexpanded!()
}

#[derive(new)]
pub struct Fill<M: Expr<Output = Matrix<Value::Output>>, Value: Expr>
where
    Value::Output: SquareType,
{
    matrix: M,
    value: Value,
}

impl<M: Expr<Output = Matrix<Value::Output>>, Value: Expr> Expr for Fill<M, Value>
where
    Value::Output: SquareType,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        CmmaExpression::Fill {
            matrix: Box::new(self.matrix.expression_untyped()),
            value: Box::new(self.value.expression_untyped()),
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

/// Module containing the expand function for [fill()].
pub mod fill {
    use super::*;

    /// Expand method of [fill()].
    pub fn expand<C: SquareType>(
        mat: impl Expr<Output = Matrix<C>>,
        value: impl Expr<Output = C>,
    ) -> impl Expr<Output = ()> {
        Fill::new(mat, value)
    }
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: SquareType, Slice: Strided + Container<Item = C>>(
    mat: &Matrix<C>,
    value: &Slice,
    stride: u32,
) {
    unexpanded!()
}

#[derive(new)]
pub struct CmmaLoad<
    T: SquareType,
    Mat: Expr<Output = Matrix<T>>,
    Slice: Expr,
    Stride: Expr<Output = u32>,
> where
    Slice::Output: Strided + Container<Item = T>,
{
    pub matrix: Mat,
    pub values: Slice,
    pub stride: Stride,
}

impl<T: SquareType, Mat: Expr<Output = Matrix<T>>, Slice: Expr, Stride: Expr<Output = u32>> Expr
    for CmmaLoad<T, Mat, Slice, Stride>
where
    Slice::Output: Strided + Container<Item = T>,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        CmmaExpression::Load {
            matrix: Box::new(self.matrix.expression_untyped()),
            values: Box::new(self.values.expression_untyped()),
            stride: Box::new(self.stride.expression_untyped()),
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

/// Module containing the expand function for [load()].
pub mod load {
    use super::*;

    /// Expand method of [load()].
    #[allow(unused_variables)]
    pub fn expand<C: SquareType, Slice: Expr>(
        mat: impl Expr<Output = Matrix<C>>,
        value: Slice,
        stride: u32,
    ) -> impl Expr<Output = ()>
    where
        Slice::Output: Strided + Container<Item = C>,
    {
        CmmaLoad::new(mat, value, stride)
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn store<C: SquareType, Slice: Strided + Container<Item = C>>(
    output: &mut Slice,
    mat: &Matrix<C>,
    stride: impl Expr<Output = u32>,
    layout: MatrixLayout,
) {
    unexpanded!()
}

#[derive(new)]
pub struct CmmaStore<
    T: SquareType,
    Mat: Expr<Output = Matrix<T>>,
    Slice: Expr,
    Stride: Expr<Output = u32>,
> where
    Slice::Output: Strided + Container<Item = T>,
{
    pub matrix: Mat,
    pub output: Slice,
    pub stride: Stride,
    pub layout: MatrixLayout,
}

impl<T: SquareType, Mat: Expr<Output = Matrix<T>>, Slice: Expr, Stride: Expr<Output = u32>> Expr
    for CmmaStore<T, Mat, Slice, Stride>
where
    Slice::Output: Strided + Container<Item = T>,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        CmmaExpression::Store {
            matrix: Box::new(self.matrix.expression_untyped()),
            out: Box::new(self.output.expression_untyped()),
            stride: Box::new(self.stride.expression_untyped()),
            layout: self.layout,
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

/// Module containing the expand function for [store()].
pub mod store {
    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<T: SquareType, Slice: Expr>(
        output: Slice,
        mat: impl Expr<Output = Matrix<T>>,
        stride: impl Expr<Output = u32>,
        layout: MatrixLayout,
    ) -> impl Expr<Output = ()>
    where
        Slice::Output: Strided + Container<Item = T>,
    {
        CmmaStore::new(mat, output, stride, layout)
    }
}

/// Execute the matrix-multiply and accumulate operation on the given [matrices](Matrix).
#[allow(unused_variables)]
pub fn execute<A: SquareType, B: SquareType, C: SquareType, D: SquareType>(
    mat_a: &Matrix<A>,
    mat_b: &Matrix<B>,
    mat_c: &Matrix<C>,
    mat_d: &Matrix<D>,
) {
    unexpanded!()
}

#[derive(new)]
pub struct CmmaExecute<
    A: SquareType,
    B: SquareType,
    C: SquareType,
    D: SquareType,
    MatA: Expr<Output = Matrix<A>>,
    MatB: Expr<Output = Matrix<B>>,
    MatC: Expr<Output = Matrix<C>>,
    MatD: Expr<Output = Matrix<D>>,
> {
    pub mat_a: MatA,
    pub mat_b: MatB,
    pub mat_c: MatC,
    pub mat_d: MatD,
}

impl<
        A: SquareType,
        B: SquareType,
        C: SquareType,
        D: SquareType,
        MatA: Expr<Output = Matrix<A>>,
        MatB: Expr<Output = Matrix<B>>,
        MatC: Expr<Output = Matrix<C>>,
        MatD: Expr<Output = Matrix<D>>,
    > Expr for CmmaExecute<A, B, C, D, MatA, MatB, MatC, MatD>
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        CmmaExpression::Execute {
            mat_a: Box::new(self.mat_a.expression_untyped()),
            mat_b: Box::new(self.mat_b.expression_untyped()),
            mat_c: Box::new(self.mat_c.expression_untyped()),
            mat_d: Box::new(self.mat_d.expression_untyped()),
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

/// Module containing the expand function for [execute()].
pub mod execute {
    use super::*;

    /// Expand method of [execute()].
    pub fn expand<
        A: SquareType,
        B: SquareType,
        C: SquareType,
        D: SquareType,
        MatA: Expr<Output = Matrix<A>>,
        MatB: Expr<Output = Matrix<B>>,
        MatC: Expr<Output = Matrix<C>>,
        MatD: Expr<Output = Matrix<D>>,
    >(
        mat_a: MatA,
        mat_b: MatB,
        mat_c: MatC,
        mat_d: MatD,
    ) -> impl Expr<Output = ()> {
        CmmaExecute::new(mat_a, mat_b, mat_c, mat_d)
    }
}
