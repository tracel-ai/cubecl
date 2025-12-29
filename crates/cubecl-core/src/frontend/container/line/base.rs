use crate as cubecl;
use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped},
    prelude::MulHi,
};
use crate::{
    ir::{Arithmetic, BinaryOperator, Instruction, Scope, Type},
    prelude::{Dot, Numeric, binary_expand_fixed_output},
    unexpanded,
};
use cubecl_ir::{Comparison, ConstantValue, ExpandElement, StorageType};
use cubecl_macros::{cube, intrinsic};
use derive_more::derive::Neg;
/// A contiguous list of elements that supports auto-vectorized operations.

#[derive(Neg)]
pub struct Line<P> {
    // Comptime lines only support 1 element.
    pub(crate) val: P,
}

type LineExpand<P> = ExpandElementTyped<Line<P>>;

impl<P: CubePrimitive> Clone for Line<P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: CubePrimitive> Eq for Line<P> {}
impl<P: CubePrimitive> Copy for Line<P> {}

/// Module that contains the implementation details of the new function.
mod new {
    use cubecl_ir::LineSize;
    use cubecl_macros::comptime_type;

    use super::*;

    #[cube]
    impl<P: CubePrimitive> Line<P> {
        /// Create a new line of size 1 using the given value.
        #[allow(unused_variables)]
        pub fn new(val: P) -> Self {
            intrinsic!(|_| {
                let elem: ExpandElementTyped<P> = val;
                elem.expand.into()
            })
        }
    }

    impl<P: CubePrimitive> Line<P> {
        /// Get the length of the current line.
        pub fn line_size(&self) -> comptime_type!(LineSize) {
            unexpanded!()
        }
    }
}

/// Module that contains the implementation details of the fill function.
mod fill {
    use crate::prelude::cast;

    use super::*;

    #[cube]
    impl<P: CubePrimitive + Into<ExpandElementTyped<P>>> Line<P> {
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
                let length = self.expand.ty.line_size();
                let output = scope.create_local(Type::new(P::as_type(scope)).line(length));

                cast::expand::<P, Line<P>>(scope, value, output.clone().into());

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
    impl<P: CubePrimitive> Line<P> {
        /// Create an empty line of the given size.
        ///
        /// Note that a line can't change in size once it's fixed.
        #[allow(unused_variables)]
        pub fn empty(#[comptime] size: usize) -> Self {
            let zero = Line::<P>::cast_from(0);
            intrinsic!(|scope| {
                // We don't declare const variables in our compilers, only mut variables.
                // So we need to create the variable as mut here.
                let var: ExpandElementTyped<Line<P>> = scope
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

    impl<P: CubePrimitive> Line<P> {
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
            unexpanded!()
        }

        /// Expand function of [size](Self::size).
        pub fn __expand_size(scope: &mut Scope, element: ExpandElementTyped<P>) -> LineSize {
            element.__expand_line_size_method(scope)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Line<P>> {
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
                impl<P: CubePrimitive> Line<P> {
                    #[doc = concat!(
                        "Return a new line with the element-wise comparison of the first line being ",
                        $comment,
                        " the second line."
                    )]
                    #[allow(unused_variables)]
                    pub fn $name(self, other: Self) -> Line<bool> {
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
    impl Line<bool> {
        /// Return a new line with the element-wise and of the lines
        #[allow(unused_variables)]
        pub fn and(self, other: Self) -> Line<bool> {
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
    impl Line<bool> {
        /// Return a new line with the element-wise and of the lines
        #[allow(unused_variables)]
        pub fn or(self, other: Self) -> Line<bool> {
            intrinsic!(|scope| binary_expand(scope, self.expand, other.expand, Operator::Or).into())
        }
    }
}

impl<P: CubePrimitive> CubeType for Line<P> {
    type ExpandType = ExpandElementTyped<Self>;
}

impl<P: CubePrimitive> CubeType for &Line<P> {
    type ExpandType = ExpandElementTyped<Line<P>>;
}

impl<P: CubePrimitive> CubeType for &mut Line<P> {
    type ExpandType = ExpandElementTyped<Line<P>>;
}

impl<P: CubePrimitive> ExpandElementIntoMut for Line<P> {
    fn elem_into_mut(scope: &mut crate::ir::Scope, elem: ExpandElement) -> ExpandElement {
        P::elem_into_mut(scope, elem)
    }
}

impl<P: CubePrimitive> CubePrimitive for Line<P> {
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

impl<N: Numeric> Dot for Line<N> {
    fn dot(self, _rhs: Self) -> Self {
        unexpanded!()
    }

    fn __expand_dot(
        scope: &mut Scope,
        lhs: ExpandElementTyped<Self>,
        rhs: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<Self> {
        let lhs: ExpandElement = lhs.into();
        let item = lhs.ty.storage_type().into();
        binary_expand_fixed_output(scope, lhs, rhs.into(), item, Arithmetic::Dot).into()
    }
}

impl<N: MulHi + CubePrimitive> MulHi for Line<N> {}
