use core::{marker::PhantomData, ops::Neg};

use crate::frontend::{CubePrimitive, CubeType, NativeAssign, NativeExpand};
use crate::ir::Scope;
use crate::{self as cubecl, prelude::*};
use cubecl_ir::{ConstantValue, ExpandValue, types::VectorType};
use cubecl_macros::{cube, intrinsic};
use num_traits::Zero;
use pliron::r#type::TypeHandle;

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
impl<P: Scalar + NegNativeExpand, N: Size> NegNativeExpand for Vector<P, N> {
    fn __expand_native_neg(scope: &Scope, this: ExpandValue) -> ExpandValue {
        P::__expand_native_neg(scope, this)
    }
}

/// Module that contains the implementation details of the new function.
mod new {
    use cubecl_ir::VectorSize;
    use cubecl_macros::comptime_type;

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

        pub fn __expand_new(scope: &Scope, val: NativeExpand<P>) -> VectorExpand<P, N> {
            broadcast_value(scope, val.read_value(scope), N::__expand_value(scope)).into()
        }
    }

    #[cube]
    impl<P: Scalar, N: Size> Vector<P, N> {
        pub fn broadcast(value: P) -> Vector<P, N> {
            Self::new(value)
        }
    }

    impl<P: Scalar, N: Size> Vector<P, N> {
        /// Get the length of the current vector.
        pub fn vector_size(&self) -> comptime_type!(VectorSize) {
            N::value()
        }
    }
}

mod components {
    use cubecl_ir::{
        dialect::vector::{
            VectorExtractDynamicOp, VectorExtractOp, VectorInsertDynamicOp, VectorInsertOp,
        },
        interfaces::TypedExt,
    };

    use super::*;

    #[cube]
    impl<P: Scalar, N: Size> Vector<P, N> {
        pub fn extract(self, #[comptime] index: usize) -> P {
            intrinsic!(|scope| {
                let this = self.read_value(scope);
                if this.vector_size(scope.ctx()) > 1 {
                    let op = VectorExtractOp::new(scope.ctx_mut(), this, index);
                    scope.register_with_result(&op).into()
                } else {
                    this.into()
                }
            })
        }

        pub fn insert(&mut self, #[comptime] index: usize, value: P) {
            intrinsic!(|scope| {
                let this = self.read_value(scope);
                let value = value.read_value(scope);
                if this.vector_size(scope.ctx()) > 1 {
                    let op = VectorInsertOp::new(scope.ctx_mut(), this, value, index);
                    let new_value = scope.register_with_result(&op).into();
                    assign::expand_element(scope, new_value, self.expand);
                } else {
                    assign::expand_element(scope, value.into(), self.expand);
                }
            })
        }

        /// Dynamically extract a value from the vector. **This is extremely slow and should only
        /// be used when there is no other option**
        pub fn extract_dynamic(self, index: usize) -> P {
            intrinsic!(|scope| {
                let this = self.read_value(scope);
                if this.vector_size(scope.ctx()) > 1 {
                    let index = index.read_value(scope);
                    let op = VectorExtractDynamicOp::new(scope.ctx_mut(), this, index);
                    scope.register_with_result(&op).into()
                } else {
                    this.into()
                }
            })
        }

        /// Dynamically inmsert a value to the vector. **This is extremely slow and should only
        /// be used when there is no other option**
        pub fn insert_dynamic(&mut self, index: usize, value: P) {
            intrinsic!(|scope| {
                let this = self.read_value(scope);
                let value = value.read_value(scope);
                if this.vector_size(scope.ctx()) > 1 {
                    let index = index.read_value(scope);
                    let op = VectorInsertDynamicOp::new(scope.ctx_mut(), this, value, index);
                    let new_value = scope.register_with_result(&op).into();
                    assign::expand_element(scope, new_value, self.expand);
                } else {
                    assign::expand_element(scope, value.into(), self.expand);
                }
            })
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
        pub fn fill(self, value: P) -> Self {
            intrinsic!(|scope| { Vector::<P, N>::__expand_cast_from(scope, value) })
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

// Implement a comparison operator define in
macro_rules! impl_vector_comparison {
    ($name:ident, $operator:ident, $comment:literal) => {
        ::paste::paste! {
            /// Module that contains the implementation details of the $name function.
            mod $name {

                use super::*;

                #[cube]
                impl<P: Scalar + CubePartialOrd, N: Size> Vector<P, N> {
                    #[doc = concat!(
                        "Return a new vector with the element-wise comparison of the first vector being ",
                        $comment,
                        " the second vector."
                    )]
                    pub fn $name(&self, other: &Self) -> Vector<bool, N> {
                        intrinsic!(|scope| {
                            let this = self.__expand_deref_method(scope).into();
                            let other = other.__expand_deref_method(scope).into();

                            P::Scalar::[<__expand_native_ $operator>](scope, this, other).into()
                        })
                    }
                }
            }
        }

    };
}

impl_vector_comparison!(equal, eq, "equal to");
impl_vector_comparison!(not_equal, ne, "not equal to");
impl_vector_comparison!(less_than, lt, "less than");
impl_vector_comparison!(greater_than, gt, "greater than");
impl_vector_comparison!(less_equal, le, "less than or equal to");
impl_vector_comparison!(greater_equal, ge, "greater than or equal to");

mod bool_and {
    use cubecl_ir::dialect::general::BoolAndOp;

    use crate::prelude::binary_expand;

    use super::*;

    #[cube]
    impl<N: Size> Vector<bool, N> {
        /// Return a new vector with the element-wise and of the vectors
        pub fn vec_and(self, other: Self) -> Vector<bool, N> {
            intrinsic!(
                |scope| binary_expand(scope, self.expand, other.expand, BoolAndOp::new).into()
            )
        }
    }
}

mod bool_or {
    use cubecl_ir::dialect::general::BoolOrOp;

    use crate::prelude::binary_expand;

    use super::*;

    #[cube]
    impl<N: Size> Vector<bool, N> {
        /// Return a new vector with the element-wise and of the vectors
        pub fn or(self, other: Self) -> Vector<bool, N> {
            intrinsic!(
                |scope| binary_expand(scope, self.expand, other.expand, BoolOrOp::new).into()
            )
        }
    }
}

impl<P: Scalar, N: Size> CubeType for Vector<P, N> {
    type ExpandType = NativeExpand<Self>;
}

impl<P: Scalar, N: Size> CubeDebug for Vector<P, N> {}

impl<P: Scalar, N: Size> NativeAssign for Vector<P, N> {
    fn elem_init_mut(scope: &Scope, elem: ExpandValue) -> ExpandValue {
        P::elem_init_mut(scope, elem)
    }
}

impl<P: Scalar, N: Size> CubePrimitive for Vector<P, N> {
    type Scalar = P;
    type Size = N;
    type WithScalar<S: Scalar> = Vector<S, N>;

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        let inner = P::__expand_as_type(scope);
        let vectorization = N::__expand_value(scope);
        if vectorization > 1 {
            VectorType::get(scope.ctx(), inner, vectorization).into()
        } else {
            inner
        }
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

#[cube]
impl<T: Int, N: Size> Vector<T, N> {
    pub fn is_multiple_of(&self, multiple: T) -> Vector<bool, N> {
        (*self % Vector::new(multiple)).equal(&Vector::zero())
    }
}
