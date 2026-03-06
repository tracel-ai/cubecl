use core::{marker::PhantomData, ops::Neg};

use crate::{
    self as cubecl,
    prelude::{FloatOps, Size},
};
use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementAssign, ExpandElementTyped},
    prelude::MulHi,
};
use crate::{
    ir::{BinaryOperator, Instruction, Scope, Type},
    prelude::Dot,
    unexpanded,
};
use cubecl_ir::{Comparison, ConstantValue, ExpandElement, StorageType};
use cubecl_macros::{cube, intrinsic};

/// A contiguous list of elements that supports auto-vectorized operations.
pub struct Line<P, N: Size> {
    // Comptime lines only support 1 element.
    pub(crate) val: P,
    pub(crate) _size: PhantomData<N>,
}

type LineExpand<P, N> = ExpandElementTyped<Line<P, N>>;

impl<P: CubePrimitive, N: Size> Clone for Line<P, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: CubePrimitive, N: Size> Eq for Line<P, N> {}
impl<P: CubePrimitive, N: Size> Copy for Line<P, N> {}
impl<P: CubePrimitive + Neg<Output = P>, N: Size> Neg for Line<P, N> {
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
    use cubecl_ir::LineSize;
    use cubecl_macros::comptime_type;

    use super::*;

    #[cube]
    impl<P: CubePrimitive, N: Size> Line<P, N> {
        /// Create a new line of size 1 using the given value.
        #[allow(unused_variables)]
        pub fn new(val: P) -> Self {
            Line::empty().fill(val)
        }
    }

    impl<P: CubePrimitive, N: Size> Line<P, N> {
        /// Get the length of the current line.
        pub fn line_size(&self) -> comptime_type!(LineSize) {
            N::value()
        }
    }
}

/// Module that contains the implementation details of the fill function.
mod fill {
    use crate::prelude::cast;

    use super::*;

    #[cube]
    impl<P: CubePrimitive, N: Size> Line<P, N> {
        /// Fill the line with the given value.
        ///
        /// If you want to fill the line with different values, consider using the index API
        /// instead.
        ///
        /// ```rust, ignore
        /// let mut line = Line::<u32>::empty(2);
        /// line[0] = 1;
        /// line[1] = 2;
        /// ```
        #[allow(unused_variables)]
        pub fn fill(self, value: P) -> Self {
            intrinsic!(|scope| {
                let length = N::__expand_value(scope);
                let output = scope.create_local(Type::new(P::as_type(scope)).line(length));

                cast::expand::<P, Line<P, N>>(scope, value, output.clone().into());

                output.into()
            })
        }
    }
}

/// Module that contains the implementation details of the empty function.
mod empty {
    use crate::prelude::Cast;

    use super::*;

    #[cube]
    impl<P: CubePrimitive, N: Size> Line<P, N> {
        /// Create an empty line of the given size.
        ///
        /// Note that a line can't change in size once it's fixed.
        #[allow(unused_variables)]
        pub fn empty() -> Self {
            let zero = Line::<P, N>::cast_from(0);
            intrinsic!(|scope| {
                let size = N::__expand_value(scope);
                // We don't declare const variables in our compilers, only mut variables.
                // So we need to create the variable as mut here.
                let var: ExpandElementTyped<Line<P, N>> = scope
                    .create_local_mut(Type::new(Self::as_type(scope)).line(size))
                    .into();
                cubecl::frontend::assign::expand(scope, zero, var.clone());
                var
            })
        }
    }
}

/// Module that contains the implementation details of the size function.
mod size {
    use cubecl_ir::LineSize;

    use super::*;

    impl<P: CubePrimitive, N: Size> Line<P, N> {
        /// Get the number of individual elements a line contains.
        ///
        /// The size is available at comptime and may be used in combination with the comptime
        /// macro.
        ///
        /// ```rust, ignore
        /// // The if statement is going to be executed at comptime.
        /// if comptime!(line.size() == 1) {
        /// }
        /// ```
        pub fn size(&self) -> LineSize {
            N::value()
        }

        /// Expand function of [size](Self::size).
        pub fn __expand_size(
            scope: &mut Scope,
            element: ExpandElementTyped<Line<P, N>>,
        ) -> LineSize {
            element.__expand_line_size_method(scope)
        }
    }

    impl<P: CubePrimitive, N: Size> ExpandElementTyped<Line<P, N>> {
        /// Comptime version of [size](Line::size).
        pub fn size(&self) -> LineSize {
            self.expand.ty.line_size()
        }

        /// Expand method of [size](Line::size).
        pub fn __expand_size_method(&self, _scope: &mut Scope) -> LineSize {
            self.size()
        }
    }
}

// Implement a comparison operator define in
macro_rules! impl_line_comparison {
    ($name:ident, $operator:ident, $comment:literal) => {
        ::paste::paste! {
            /// Module that contains the implementation details of the $name function.
            mod $name {

                use super::*;

                #[cube]
                impl<P: CubePrimitive, N: Size> Line<P, N> {
                    #[doc = concat!(
                        "Return a new line with the element-wise comparison of the first line being ",
                        $comment,
                        " the second line."
                    )]
                    #[allow(unused_variables)]
                    pub fn $name(self, other: Self) -> Line<bool, N> {
                        intrinsic!(|scope| {
                            let size = self.expand.ty.line_size();
                            let lhs = self.expand.into();
                            let rhs = other.expand.into();

                            let output = scope.create_local_mut(Type::new(bool::as_type(scope)).line(size));

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

impl_line_comparison!(equal, Equal, "equal to");
impl_line_comparison!(not_equal, NotEqual, "not equal to");
impl_line_comparison!(less_than, Lower, "less than");
impl_line_comparison!(greater_than, Greater, "greater than");
impl_line_comparison!(less_equal, LowerEqual, "less than or equal to");
impl_line_comparison!(greater_equal, GreaterEqual, "greater than or equal to");

mod bool_and {
    use cubecl_ir::Operator;

    use crate::prelude::binary_expand;

    use super::*;

    #[cube]
    impl<N: Size> Line<bool, N> {
        /// Return a new line with the element-wise and of the lines
        #[allow(unused_variables)]
        pub fn and(self, other: Self) -> Line<bool, N> {
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
    impl<N: Size> Line<bool, N> {
        /// Return a new line with the element-wise and of the lines
        #[allow(unused_variables)]
        pub fn or(self, other: Self) -> Line<bool, N> {
            intrinsic!(|scope| binary_expand(scope, self.expand, other.expand, Operator::Or).into())
        }
    }
}

impl<P: CubePrimitive, N: Size> CubeType for Line<P, N> {
    type ExpandType = ExpandElementTyped<Self>;
}

impl<P: CubePrimitive, N: Size> CubeType for &Line<P, N> {
    type ExpandType = ExpandElementTyped<Line<P, N>>;
}

impl<P: CubePrimitive, N: Size> CubeType for &mut Line<P, N> {
    type ExpandType = ExpandElementTyped<Line<P, N>>;
}

impl<P: CubePrimitive, N: Size> ExpandElementAssign for Line<P, N> {
    fn elem_init_mut(scope: &mut crate::ir::Scope, elem: ExpandElement) -> ExpandElement {
        P::elem_init_mut(scope, elem)
    }
}

impl<P: CubePrimitive, N: Size> CubePrimitive for Line<P, N> {
    fn as_type(scope: &Scope) -> StorageType {
        P::as_type(scope)
    }

    fn as_type_native() -> Option<StorageType> {
        P::as_type_native()
    }

    fn size() -> Option<usize> {
        P::size()
    }

    fn from_const_value(value: ConstantValue) -> Self {
        Self::new(P::from_const_value(value))
    }
}

impl<T: Dot + CubePrimitive, N: Size> Dot for Line<T, N> {}
impl<T: MulHi + CubePrimitive, N: Size> MulHi for Line<T, N> {}
impl<T: FloatOps + CubePrimitive, N: Size> FloatOps for Line<T, N> {}
