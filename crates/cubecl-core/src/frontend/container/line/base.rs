use std::num::NonZero;

use crate as cubecl;
use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped},
    prelude::MulHi,
};
use crate::{
    ir::{Arithmetic, BinaryOperator, Elem, Instruction, Item, Scope},
    prelude::{Dot, Numeric, binary_expand_fixed_output},
    unexpanded,
};
use cubecl_ir::{Comparison, ExpandElement};
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
                let length = self.expand.item.vectorization;
                let output = scope.create_local(Item::vectorized(P::as_elem(scope), length));

                cast::expand::<P>(scope, value, output.clone().into());

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
        pub fn empty(#[comptime] size: u32) -> Self {
            let zero = Line::<P>::cast_from(0);
            intrinsic!(|scope| {
                let length = NonZero::new(size as u8);
                // We don't declare const variables in our compilers, only mut variables.
                // So we need to create the variable as mut here.
                let var: ExpandElementTyped<Line<P>> = scope
                    .create_local_mut(Item::vectorized(Self::as_elem(scope), length))
                    .into();
                cubecl::frontend::assign::expand(scope, zero, var.clone());
                var
            })
        }
    }
}

/// Module that contains the implementation details of the size function.
mod size {
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
        pub fn size(&self) -> u32 {
            unexpanded!()
        }

        /// Expand function of [size](Self::size).
        pub fn __expand_size(scope: &mut Scope, element: ExpandElementTyped<P>) -> u32 {
            element.__expand_vectorization_factor_method(scope)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Line<P>> {
        /// Comptime version of [size](Line::size).
        pub fn size(&self) -> u32 {
            self.expand
                .item
                .vectorization
                .unwrap_or(NonZero::new(1).unwrap())
                .get() as u32
        }

        /// Expand method of [size](Line::size).
        pub fn __expand_size_method(&self, _scope: &mut Scope) -> u32 {
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
                            let size = self.expand.item.vectorization;
                            let lhs = self.expand.into();
                            let rhs = other.expand.into();

                            let output = scope.create_local_mut(Item::vectorized(bool::as_elem(scope), size));

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

impl<P: CubePrimitive> ExpandElementIntoMut for Line<P> {
    fn elem_into_mut(scope: &mut crate::ir::Scope, elem: ExpandElement) -> ExpandElement {
        P::elem_into_mut(scope, elem)
    }
}

impl<P: CubePrimitive> CubePrimitive for Line<P> {
    fn as_elem(scope: &Scope) -> Elem {
        P::as_elem(scope)
    }

    fn as_elem_native() -> Option<Elem> {
        P::as_elem_native()
    }

    fn size() -> Option<usize> {
        P::size()
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
        let mut item = lhs.item;
        item.vectorization = None;
        binary_expand_fixed_output(scope, lhs, rhs.into(), item, Arithmetic::Dot).into()
    }
}

impl<N: MulHi + CubePrimitive> MulHi for Line<N> {}
