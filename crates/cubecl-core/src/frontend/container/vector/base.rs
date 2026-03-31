use core::{marker::PhantomData, ops::Neg};

use crate::frontend::{CubePrimitive, CubeType, NativeAssign, NativeExpand};
use crate::ir::{BinaryOperator, Instruction, Scope, Type};
use crate::{self as cubecl, prelude::*};
use cubecl_ir::{Comparison, ConstantValue, ManagedVariable};
use cubecl_macros::{cube, intrinsic};

/// A contiguous list of elements that supports auto-vectorized operations.
#[derive(Debug)]
pub struct Vector<P: Scalar, N: Size> {
    // Comptime vectors only support 1 element.
    pub(crate) val: P,
    pub(crate) _size: PhantomData<N>,
}

type VectorExpand<P, N> = NativeExpand<Vector<P, N>>;

impl<P: Scalar, N: Size> Clone for Vector<P, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: Scalar, N: Size> Eq for Vector<P, N> {}
impl<P: Scalar, N: Size> Copy for Vector<P, N> {}
impl<P: Scalar + Neg<Output = P>, N: Size> Neg for Vector<P, N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            val: -self.val,
            _size: PhantomData,
        }
    }
}

/// Module that contains the implementation details of the new function.
mod new {
    use cubecl_ir::VectorSize;
    use cubecl_macros::comptime_type;

    use crate::prelude::Cast;

    use super::*;

    impl<P: Scalar, N: Size> Vector<P, N> {
        /// Create a new vector of size 1 using the given value.
        #[allow(unused_variables)]
        pub fn new(val: P) -> Self {
            Self {
                val,
                _size: PhantomData,
            }
        }

        pub fn __expand_new(scope: &mut Scope, val: NativeExpand<P>) -> VectorExpand<P, N> {
            Vector::<P, N>::__expand_cast_from(scope, val)
        }
    }

    impl<P: Scalar, N: Size> Vector<P, N> {
        /// Get the length of the current vector.
        pub fn vector_size(&self) -> comptime_type!(VectorSize) {
            N::value()
        }
    }
}

mod numeric {
    use super::*;

    #[cube]
    impl<P: Numeric, N: Size> Vector<P, N> {
        pub fn min_value() -> Self {
            Self::new(P::min_value())
        }
        pub fn max_value() -> Self {
            Self::new(P::max_value())
        }

        /// Create a new constant numeric.
        ///
        /// Note: since this must work for both integer and float
        /// only the less expressive of both can be created (int)
        /// If a number with decimals is needed, use `Float::new`.
        ///
        /// This method panics when unexpanded. For creating an element
        /// with a val, use the new method of the sub type.
        pub fn from_int(val: i64) -> Self {
            Self::new(P::from_int(val))
        }
    }
}

/// Module that contains the implementation details of the fill function.
mod fill {
    use crate::prelude::cast;

    use super::*;

    #[cube]
    impl<P: Scalar, N: Size> Vector<P, N> {
        /// Fill the vector with the given value.
        ///
        /// If you want to fill the vector with different values, consider using the index API
        /// instead.
        ///
        /// ```rust, ignore
        /// let mut vector = Vector::<u32>::empty(2);
        /// vector[0] = 1;
        /// vector[1] = 2;
        /// ```
        #[allow(unused_variables)]
        pub fn fill(self, value: P) -> Self {
            intrinsic!(|scope| {
                let output = scope.create_local(Vector::<P, N>::as_type(scope));

                cast::expand::<P, Vector<P, N>>(scope, value, output.clone().into());

                output.into()
            })
        }
    }
}

/// Module that contains the implementation details of the empty function.
mod empty {
    use bytemuck::Zeroable;

    use super::*;

    #[cube]
    impl<P: Scalar, N: Size> Vector<P, N> {
        pub fn empty() -> Self {
            intrinsic!(|scope| {
                let value = Self::__expand_default(scope);
                value.into_mut(scope)
            })
        }
    }

    #[cube]
    impl<P: Scalar + Zeroable, N: Size> Vector<P, N> {
        pub fn zeroed() -> Self {
            intrinsic!(|scope| {
                let zeroed = P::zeroed().__expand_runtime_method(scope);
                Self::__expand_cast_from(scope, zeroed)
            })
        }
    }
}

/// Module that contains the implementation details of the size function.
mod size {
    use cubecl_ir::VectorSize;

    use super::*;

    impl<P: Scalar, N: Size> Vector<P, N> {
        /// Get the number of individual elements a vector contains.
        ///
        /// The size is available at comptime and may be used in combination with the comptime
        /// macro.
        ///
        /// ```rust, ignore
        /// // The if statement is going to be executed at comptime.
        /// if comptime!(vector.size() == 1) {
        /// }
        /// ```
        pub fn size(&self) -> VectorSize {
            N::value()
        }

        /// Expand function of [size](Self::size).
        pub fn __expand_size(scope: &mut Scope, element: NativeExpand<Vector<P, N>>) -> VectorSize {
            element.__expand_vector_size_method(scope)
        }
    }

    impl<P: Scalar, N: Size> NativeExpand<Vector<P, N>> {
        /// Comptime version of [size](Vector::size).
        pub fn size(&self) -> VectorSize {
            self.expand.ty.vector_size()
        }

        /// Expand method of [size](Vector::size).
        pub fn __expand_size_method(&self, _scope: &mut Scope) -> VectorSize {
            self.size()
        }
    }
}

// Implement a comparison operator define in
macro_rules! impl_vector_comparison {
    ($name:ident, $operator:ident, $comment:literal) => {
        ::paste::paste! {
            /// Module that contains the implementation details of the $name function.
            mod $name {

                use super::*;

                #[cube]
                impl<P: Scalar, N: Size> Vector<P, N> {
                    #[doc = concat!(
                        "Return a new vector with the element-wise comparison of the first vector being ",
                        $comment,
                        " the second vector."
                    )]
                    #[allow(unused_variables)]
                    pub fn $name(self, other: Self) -> Vector<bool, N> {
                        intrinsic!(|scope| {
                            let size = self.expand.ty.vector_size();
                            let lhs = self.expand.into();
                            let rhs = other.expand.into();

                            let output = scope.create_local_mut(Vector::<bool, N>::as_type(scope));

                            scope.register(Instruction::new(
                                Comparison::$operator(BinaryOperator { lhs, rhs }),
                                output.clone().into(),
                            ));

                            output.into()
                        })
                    }
                }
            }
        }

    };
}

impl_vector_comparison!(equal, Equal, "equal to");
impl_vector_comparison!(not_equal, NotEqual, "not equal to");
impl_vector_comparison!(less_than, Lower, "less than");
impl_vector_comparison!(greater_than, Greater, "greater than");
impl_vector_comparison!(less_equal, LowerEqual, "less than or equal to");
impl_vector_comparison!(greater_equal, GreaterEqual, "greater than or equal to");

mod bool_and {
    use cubecl_ir::Operator;

    use crate::prelude::binary_expand;

    use super::*;

    #[cube]
    impl<N: Size> Vector<bool, N> {
        /// Return a new vector with the element-wise and of the vectors
        #[allow(unused_variables)]
        pub fn and(self, other: Self) -> Vector<bool, N> {
            intrinsic!(
                |scope| binary_expand(scope, self.expand, other.expand, Operator::And).into()
            )
        }
    }
}

mod bool_or {
    use cubecl_ir::Operator;

    use crate::prelude::binary_expand;

    use super::*;

    #[cube]
    impl<N: Size> Vector<bool, N> {
        /// Return a new vector with the element-wise and of the vectors
        #[allow(unused_variables)]
        pub fn or(self, other: Self) -> Vector<bool, N> {
            intrinsic!(|scope| binary_expand(scope, self.expand, other.expand, Operator::Or).into())
        }
    }
}

impl<P: Scalar, N: Size> CubeType for Vector<P, N> {
    type ExpandType = NativeExpand<Self>;
}

impl<P: Scalar, N: Size> CubeType for &Vector<P, N> {
    type ExpandType = NativeExpand<Vector<P, N>>;
}

impl<P: Scalar, N: Size> CubeType for &mut Vector<P, N> {
    type ExpandType = NativeExpand<Vector<P, N>>;
}

impl<P: Scalar, N: Size> NativeAssign for Vector<P, N> {
    fn elem_init_mut(scope: &mut crate::ir::Scope, elem: ManagedVariable) -> ManagedVariable {
        P::elem_init_mut(scope, elem)
    }
}

impl<P: Scalar, N: Size> CubePrimitive for Vector<P, N> {
    type Scalar = P;
    type Size = N;
    type WithScalar<S: Scalar> = Vector<S, N>;

    fn as_type(scope: &Scope) -> Type {
        P::as_type(scope).with_vector_size(N::__expand_value(scope))
    }

    fn as_type_native() -> Option<Type> {
        P::as_type_native().and_then(|ty| {
            let vector_size = N::try_value_const()?;
            Some(ty.with_vector_size(vector_size))
        })
    }

    fn from_const_value(value: ConstantValue) -> Self {
        Self::new(P::from_const_value(value))
    }
}

impl<T: Dot + Scalar, N: Size> Dot for Vector<T, N> {}
impl<T: MulHi + Scalar, N: Size> MulHi for Vector<T, N> {}
impl<T: FloatOps + Scalar, N: Size> FloatOps for Vector<T, N> {}
impl<T: Hypot + Scalar, N: Size> Hypot for Vector<T, N> {}
impl<T: Rhypot + Scalar, N: Size> Rhypot for Vector<T, N> {}
