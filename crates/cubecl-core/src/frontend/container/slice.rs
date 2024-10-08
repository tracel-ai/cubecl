use std::marker::PhantomData;

use crate::frontend::{
    Array, CubePrimitive, CubeType, ExpandElement, ExpandElementTyped, Init, SharedMemory,
    SizedContainer,
};
use crate::{
    frontend::indexation::Index,
    frontend::Tensor,
    ir::{self, Operator},
    prelude::CubeContext,
    unexpanded,
};

use super::Line;

/// A read-only contiguous list of elements
pub struct Slice<'a, E> {
    _e: PhantomData<E>,
    _l: &'a (),
}

/// A read-write contiguous list of elements.
pub struct SliceMut<'a, E> {
    _e: PhantomData<E>,
    _l: &'a mut (),
}

mod metadata {
    use super::*;

    impl<'a, E> Slice<'a, E> {
        /// Get the length of the slice.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the same slice, but with lines of length 1.
        pub fn as_aligned(&self) -> Slice<'a, Line<E>>
        where
            E: CubePrimitive,
        {
            unexpanded!()
        }
    }

    impl<'a, E> SliceMut<'a, E> {
        /// Get the length of the slice.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the same slice, but with lines of length 1.
        pub fn as_aligned(&self) -> SliceMut<'a, Line<E>>
        where
            E: CubePrimitive,
        {
            unexpanded!()
        }
    }

    impl<'a, C: CubeType> ExpandElementTyped<Slice<'a, C>> {
        // Expand method of [len](Slice::len).
        pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(context)
        }

        // Expand method of [len](Slice::as_aligned).
        pub fn __expand_as_aligned_method(
            self,
            _context: &mut CubeContext,
        ) -> ExpandElementTyped<Slice<'a, Line<C>>>
        where
            C: CubePrimitive,
        {
            self.expand.into()
        }
    }

    impl<'a, C: CubeType> ExpandElementTyped<SliceMut<'a, C>> {
        // Expand method of [len](SliceMut::len).
        pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(context)
        }

        // Expand method of [len](SliceMut::as_aligned).
        pub fn __expand_as_aligned_method(
            self,
            _context: &mut CubeContext,
        ) -> ExpandElementTyped<SliceMut<'a, Line<C>>>
        where
            C: CubePrimitive,
        {
            self.expand.into()
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use crate::{
        ir::{BinaryOperator, Operator},
        prelude::{CubeIndex, CubeIndexMut},
    };

    use super::*;

    impl<'a, E: CubePrimitive> Slice<'a, E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_unchecked<I: Index>(&self, _i: I) -> &E
        where
            Self: CubeIndex<I>,
        {
            unexpanded!()
        }
    }

    impl<'a, E: CubePrimitive> SliceMut<'a, E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_unchecked<I: Index>(&self, _i: I) -> &E
        where
            Self: CubeIndex<I>,
        {
            unexpanded!()
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_assign_unchecked<I: Index>(&mut self, _i: I, _value: E)
        where
            Self: CubeIndexMut<I>,
        {
            unexpanded!()
        }
    }

    impl<'a, E: CubePrimitive> ExpandElementTyped<Slice<'a, E>> {
        pub fn __expand_index_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = context.create_local_binding(self.expand.item());
            context.register(Operator::UncheckedIndex(BinaryOperator {
                out: *out,
                lhs: *self.expand,
                rhs: i.expand.consume(),
            }));
            out.into()
        }
    }

    impl<'a, E: CubePrimitive> ExpandElementTyped<SliceMut<'a, E>> {
        pub fn __expand_index_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = context.create_local_binding(self.expand.item());
            context.register(Operator::UncheckedIndex(BinaryOperator {
                out: *out,
                lhs: *self.expand,
                rhs: i.expand.consume(),
            }));
            out.into()
        }

        pub fn __expand_index_assign_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
            value: ExpandElementTyped<E>,
        ) {
            context.register(Operator::UncheckedIndexAssign(BinaryOperator {
                out: *self.expand,
                lhs: i.expand.consume(),
                rhs: value.expand.consume(),
            }));
        }
    }
}

impl<'a, E: CubeType> CubeType for Slice<'a, E> {
    type ExpandType = ExpandElementTyped<Slice<'static, E>>;
}

impl<'a, C: CubeType> Init for ExpandElementTyped<Slice<'a, C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<'a, E: CubeType> CubeType for SliceMut<'a, E> {
    type ExpandType = ExpandElementTyped<SliceMut<'static, E>>;
}

impl<'a, C: CubeType> Init for ExpandElementTyped<SliceMut<'a, C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<'a, C: CubeType<ExpandType = ExpandElementTyped<C>>> SizedContainer for Slice<'a, C> {
    type Item = C;
}

impl<'a, T: CubeType> Iterator for Slice<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}
impl<'a, T: CubeType> Iterator for &Slice<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

pub trait SliceOperator<E: CubeType>: CubeType<ExpandType = Self::Expand> {
    type Expand: SliceOperatorExpand<E>;

    /// Return a read-only view of all elements comprise between the start and end index.
    #[allow(unused_variables)]
    fn slice<Start: Index, End: Index>(&self, start: Start, end: End) -> &'_ Slice<'_, E> {
        unexpanded!()
    }
    /// Expand function of [SliceOperator::slice].
    fn __expand_slice(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.__expand_slice_method(context, start, end)
    }

    /// Return a read-write view of all elements comprise between the start and end index.
    #[allow(unused_variables)]
    fn slice_mut<Start: Index, End: Index>(
        &mut self,
        start: Start,
        end: End,
    ) -> &'_ mut SliceMut<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::slice_mut].
    fn __expand_slice_mut(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.__expand_slice_mut_method(context, start, end)
    }

    /// Return a read-write view of all elements comprise between the start and end index.
    ///
    /// # Warning
    ///
    /// Ignore the multiple borrow rule.
    #[allow(unused_variables)]
    fn slice_mut_unsafe<Start: Index, End: Index>(
        &self,
        start: Start,
        end: End,
    ) -> &'_ mut SliceMut<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::slice_mut_unsafe].
    fn __expand_slice_mut_unsafe(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.__expand_slice_mut_unsafe_method(context, start, end)
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn as_slice(&self) -> &'_ Slice<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::as_slice].
    fn __expand_as_slice(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.__expand_as_slice_method(context)
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables)]
    fn as_slice_mut(&mut self) -> &'_ mut SliceMut<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::as_slice_mut].
    fn __expand_as_slice_mut(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.__expand_as_slice_mut_method(context)
    }

    /// Reinterprete the current type as a read-write slice.
    ///
    /// # Warning
    ///
    /// Ignore the multiple borrow rule.
    #[allow(unused_variables)]
    fn as_slice_mut_unsafe(&self) -> &'_ mut SliceMut<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::as_slice_mut_unsafe].
    fn __expand_as_slice_mut_unsafe(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.__expand_as_slice_mut_unsafe_method(context)
    }
}

pub trait SliceOperatorExpand<E: CubeType>: Into<ExpandElement> + Clone {
    fn slice_base<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElement;

    fn __expand_slice_method(
        &self,
        context: &mut CubeContext,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn __expand_slice_mut_method(
        &self,
        context: &mut CubeContext,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn __expand_slice_mut_unsafe_method(
        &self,
        context: &mut CubeContext,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn __expand_as_slice_method(
        &self,
        _context: &mut CubeContext,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }

    fn __expand_as_slice_mut_unsafe_method(
        &self,
        context: &mut CubeContext,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        self.__expand_as_slice_mut_method(context)
    }

    fn __expand_as_slice_mut_method(
        &self,
        _context: &mut CubeContext,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }
}

macro_rules! slice_op {
    ($type:ident) => {
        impl<E: CubePrimitive> SliceOperator<E> for $type<E> {
            type Expand = ExpandElementTyped<$type<E>>;
        }

        impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<E>> {
            fn slice_base<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElement {
                slice_expand(context, self.clone(), start, end)
            }
        }
    };
    (slice $type:ident) => {
        impl<'a, E: CubePrimitive> SliceOperator<E> for $type<'a, E> {
            type Expand = ExpandElementTyped<$type<'static, E>>;
        }

        impl<'a, E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<'a, E>> {
            fn slice_base<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElement {
                slice_expand(context, self.clone(), start, end)
            }
        }
    };
}

slice_op!(Array);
slice_op!(Tensor);
slice_op!(SharedMemory);
slice_op!(slice Slice);
slice_op!(slice SliceMut);

pub fn slice_expand<I: Into<ExpandElement>, S1: Index, S2: Index>(
    context: &mut CubeContext,
    input: I,
    start: S1,
    end: S2, // Todo use it to get the length.
) -> ExpandElement {
    let input = input.into();
    let out = context.create_slice(input.item());

    context.register(Operator::Slice(ir::SliceOperator {
        input: *input,
        start: start.value(),
        end: end.value(),
        out: *out,
    }));

    out
}
