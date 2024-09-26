use std::num::NonZero;

use crate::{
    ir::{ConstantScalarValue, Item},
    prelude::{assign, CubeContext, ExpandElement},
    unexpanded,
};

use crate::frontend::{
    CubePrimitive, CubeType, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime,
};

/// A contiguous list of elements that supports auto-vectorized operations.
#[derive(Clone, Copy, Eq)]
pub struct Line<P: CubePrimitive> {
    // Comptime lines only support 1 element.
    pub(crate) val: P,
}

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
            let length = self.expand.item().vectorization;
            let output = context.create_local_binding(Item::vectorized(P::as_elem(), length));

            assign::expand::<P>(context, value, output.clone().into());

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
                    ConstantScalarValue::UInt(val) => NonZero::new(val as u8),
                    ConstantScalarValue::Bool(_) => None,
                },
                None => None,
            };
            context
                .create_local_variable(Item::vectorized(Self::as_elem(), length))
                .into()
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
        pub fn __expand_size(context: &mut CubeContext, element: ExpandElementTyped<P>) -> u32 {
            element.__expand_vectorization_factor_method(context)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Line<P>> {
        /// Comptime version of [size](Line::size).
        pub fn size(&self) -> u32 {
            self.expand
                .item()
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
    fn as_elem() -> crate::ir::Elem {
        P::as_elem()
    }
}
