use std::num::NonZero;

use crate::{
    ir::{BinaryOperator, ConstantScalarValue, Elem, Instruction, Item, Operator},
    prelude::{binary_expand_fixed_output, CubeContext, Dot, ExpandElement, Numeric},
    unexpanded,
};

use crate::frontend::{
    CubePrimitive, CubeType, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime,
};

/// A contiguous list of elements that supports auto-vectorized operations.
pub struct Line<P> {
    // Comptime lines only support 1 element.
    pub(crate) val: P,
}

impl<P: CubePrimitive> Clone for Line<P> {
    fn clone(&self) -> Self {
        Self {
            val: self.val.clone(),
        }
    }
}
impl<P: CubePrimitive> Eq for Line<P> {}
impl<P: CubePrimitive> Copy for Line<P> {}

/// Module that contains the implementation details of the new function.
mod new {
    use super::*;

    impl<P: CubePrimitive> Line<P> {
        /// Create a new line of size 1 using the given value.
        pub fn new(val: P) -> Self {
            Self { val }
        }

        /// Expand function of [Self::new].
        pub fn __expand_new(
            _context: &mut CubeContext,
            val: P::ExpandType,
        ) -> ExpandElementTyped<Self> {
            let elem: ExpandElementTyped<P> = val;
            elem.expand.into()
        }
    }
}

/// Module that contains the implementation details of the fill function.
mod fill {
    use crate::prelude::cast;

    use super::*;

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
        pub fn fill(mut self, value: P) -> Self {
            self.val = value;
            self
        }

        /// Expand function of [fill](Self::fill).
        pub fn __expand_fill(
            context: &mut CubeContext,
            line: ExpandElementTyped<Self>,
            value: ExpandElementTyped<P>,
        ) -> ExpandElementTyped<Self> {
            line.__expand_fill_method(context, value)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Line<P>> {
        /// Expand method of [fill](Line::fill).
        pub fn __expand_fill_method(
            self,
            context: &mut CubeContext,
            value: ExpandElementTyped<P>,
        ) -> Self {
            let length = self.expand.item.vectorization;
            let output =
                context.create_local_binding(Item::vectorized(P::as_elem(context), length));

            cast::expand::<P>(context, value, output.clone().into());

            output.into()
        }
    }
}

/// Module that contains the implementation details of the empty function.
mod empty {
    use super::*;

    impl<P: CubePrimitive + Into<ExpandElementTyped<P>>> Line<P> {
        /// Create an empty line of the given size.
        ///
        /// Note that a line can't change in size once it's fixed.
        #[allow(unused_variables)]
        pub fn empty(size: u32) -> Self {
            unexpanded!()
        }

        /// Expand function of [empty](Self::empty).
        pub fn __expand_empty(
            context: &mut CubeContext,
            length: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<Self> {
            let length = match length.expand.as_const() {
                Some(val) => match val {
                    ConstantScalarValue::Int(val, _) => NonZero::new(val)
                        .map(|val| val.get() as u8)
                        .map(|val| NonZero::new(val).unwrap()),
                    ConstantScalarValue::Float(val, _) => NonZero::new(val as i64)
                        .map(|val| val.get() as u8)
                        .map(|val| NonZero::new(val).unwrap()),
                    ConstantScalarValue::UInt(val, _) => NonZero::new(val as u8),
                    ConstantScalarValue::Bool(_) => None,
                },
                None => None,
            };
            context
                .create_local_variable(Item::vectorized(Self::as_elem(context), length))
                .into()
        }
    }
}

// impl<E: CubePrimitive> TypeMap for Line<E> {
//     type ExpandGeneric<const POS: u8> = Line<E::ExpandGeneric<POS>>;
//
//     fn register<const POS: u8>(context: &mut CubeContext) {
//         let elem = Self::as_elem(context);
//         context.register_type::<Self::ExpandGeneric<POS>>(elem);
//     }
// }

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
        pub fn __expand_size(context: &mut CubeContext, element: ExpandElementTyped<P>) -> u32 {
            element.__expand_vectorization_factor_method(context)
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
        pub fn __expand_size_method(&self, _context: &mut CubeContext) -> u32 {
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

                impl<P: CubePrimitive> Line<P> {
                    #[doc = concat!(
                        "Return a new line with the element-wise comparison of the first line being ",
                        $comment,
                        " the second line."
                    )]
                    pub fn $name(self, _other: Self) -> Line<bool> {
                        unexpanded!()
                    }

                    /// Expand function of [$name](Self::$name).
                    pub fn [< __expand_ $name >](
                        context: &mut CubeContext,
                        lhs: ExpandElementTyped<Self>,
                        rhs: ExpandElementTyped<Self>,
                    ) -> ExpandElementTyped<Line<bool>> {
                        lhs.[< __expand_ $name _method >](context, rhs)
                    }
                }

                impl<P: CubePrimitive> ExpandElementTyped<Line<P>> {
                    /// Expand method of [equal](Line::equal).
                    pub fn [< __expand_ $name _method >](
                        self,
                        context: &mut CubeContext,
                        rhs: Self,
                    ) -> ExpandElementTyped<Line<bool>> {
                        let size = self.expand.item.vectorization;
                        let lhs = self.expand.into();
                        let rhs = rhs.expand.into();

                        let output = context.create_local_binding(Item::vectorized(bool::as_elem(context), size));

                        context.register(Instruction::new(
                            Operator::$operator(BinaryOperator { lhs, rhs }),
                            output.clone().into(),
                        ));

                        output.into()
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

impl<P: CubePrimitive> CubeType for Line<P> {
    type ExpandType = ExpandElementTyped<Self>;
}

impl<P: CubePrimitive> ExpandElementBaseInit for Line<P> {
    fn init_elem(context: &mut crate::prelude::CubeContext, elem: ExpandElement) -> ExpandElement {
        P::init_elem(context, elem)
    }
}

impl<P: CubePrimitive> IntoRuntime for Line<P> {
    fn __expand_runtime_method(
        self,
        context: &mut crate::prelude::CubeContext,
    ) -> Self::ExpandType {
        self.val.__expand_runtime_method(context).expand.into()
    }
}

impl<P: CubePrimitive> CubePrimitive for Line<P> {
    fn as_elem(context: &CubeContext) -> Elem {
        P::as_elem(context)
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
        context: &mut CubeContext,
        lhs: ExpandElementTyped<Self>,
        rhs: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<Self> {
        let lhs: ExpandElement = lhs.into();
        let mut item = lhs.item;
        item.vectorization = None;
        binary_expand_fixed_output(context, lhs, rhs.into(), item, Operator::Dot).into()
    }
}
