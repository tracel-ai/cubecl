use super::{
    Line, ReadOnly, ReadWrite, SharedMemory, SliceOriginExpand, SliceV2, SliceV2Expand,
    SliceVisibility, Tensor,
};
use crate::{
    frontend::{Array, CubePrimitive, CubeType, ExpandElementTyped, Init, indexation::Index},
    ir::{Instruction, Scope},
    prelude::CubeDebug,
    unexpanded,
};
use cubecl_common::tf32;
use cubecl_ir::Operator;

/// A read-only contiguous list of elements
///
/// # Safety
///
/// Since data can't be deallocated during kernel execution, this is safe.
pub type Slice<E, IO = ReadOnly> = SliceV2<E, IO>;
pub type SliceMut<E> = SliceV2<E, ReadWrite>;

#[allow(unused)]
mod metadata {
    use core::num::NonZero;

    use cubecl_ir::{Elem, FloatKind, Item, NonSemantic};

    use crate::prelude::cube_comment;

    use super::*;

    impl<E: CubePrimitive> Slice<E> {
        /// Returns the same slice, but with lines of length 1.
        pub fn into_lined(&self) -> Slice<Line<E>>
        where
            E: CubePrimitive,
        {
            unexpanded!()
        }
        /// Try to cast the slice to the given type and panic if the type isn't the same.
        ///
        /// This function should only be used to satisfy the Rust type system, when two generic
        /// types are supposed to be the same.
        pub fn try_cast_unchecked<T>(&self) -> Slice<T>
        where
            E: CubePrimitive,
            T: CubePrimitive,
        {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> Slice<Line<E>> {
        /// Return a new Slice with updated line_size. This doesn't copy or move the data,
        /// it simply reinterpret how they are loaded and stored in memory.
        ///
        /// # Warning
        ///
        /// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
        pub fn with_line_size(&self, line_size: u32) -> Slice<Line<E>> {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> SliceMut<E> {
        /// Try to cast the slice to the given type and panic if the type isn't the same.
        ///
        /// This function should only be used to satisfy the Rust type system, when two generic
        /// types are supposed to be the same.
        pub fn try_cast_unchecked<T>(&self) -> SliceMut<T>
        where
            E: CubePrimitive,
            T: CubePrimitive,
        {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> SliceMut<Line<E>> {
        /// Return a new SliceMut with updated line_size. This doesn't copy or move the data,
        /// it simply reinterpret how they are loaded and stored in memory.
        ///
        /// # Warning
        ///
        /// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
        pub fn with_line_size(&self, line_size: u32) -> SliceMut<Line<E>> {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<Slice<E>> {
        // Expand method of [len](Slice::len).
        pub fn __expand_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(scope)
        }

        /// Expand method of [len](Slice::into_lined).
        pub fn __expand_into_lined_method(
            self,
            _scope: &mut Scope,
        ) -> ExpandElementTyped<Slice<Line<E>>>
        where
            E: CubePrimitive,
        {
            self.expand.into()
        }

        /// Expand method of [try_cast_unchecked](Slice::try_cast_unchecked).
        pub fn __expand_try_cast_unchecked_method<T>(
            self,
            scope: &mut Scope,
        ) -> ExpandElementTyped<Slice<T>>
        where
            E: CubePrimitive,
            T: CubePrimitive,
        {
            if T::as_elem(scope) != E::as_elem(scope) && !is_tf32::<E, T>(scope) {
                let elems = [T::as_elem(scope), E::as_elem(scope)];
                let is_flex32_cast = elems.contains(&Elem::Float(FloatKind::F32))
                    && elems.contains(&Elem::Float(FloatKind::Flex32));

                if !is_flex32_cast {
                    panic!(
                        "Try cast unchecked should only be used to satisfy the rust type system."
                    )
                }
            }

            self.expand.into()
        }

        pub fn __expand_clone_method(self, _scope: &mut Scope) -> ExpandElementTyped<Slice<Line<E>>>
        where
            E: CubePrimitive,
        {
            self.expand.into()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<Slice<Line<E>>> {
        /// Expand method of [with_line_size](Slice::with_line_size).
        pub fn __expand_with_line_size_method(
            self,
            scope: &mut Scope,
            line_size: u32,
        ) -> ExpandElementTyped<Slice<Line<E>>>
        where
            E: CubePrimitive,
        {
            let input = self.clone().into_variable();
            let mut item = input.item;

            if line_size as u8 == item.vectorization.unwrap_or(NonZero::new(1).unwrap()).get() {
                return self;
            }

            item.vectorization = NonZero::new(line_size as u8);
            let out = scope.create_slice(item);

            scope.register(Instruction::new(
                Operator::ReinterpretSlice(cubecl_ir::ReinterpretSliceOperator {
                    input,
                    line_size,
                }),
                *out,
            ));

            out.into()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<SliceMut<E>> {
        // Expand method of [len](SliceMut::len).
        pub fn __expand_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(scope)
        }

        /// Expand method of [len](SliceMut::into_lined).
        pub fn __expand_into_lined_method(
            self,
            _scope: &mut Scope,
        ) -> ExpandElementTyped<SliceMut<Line<E>>>
        where
            E: CubePrimitive,
        {
            self.expand.into()
        }

        /// Expand method of [try_cast_unchecked](Slice::try_cast_unchecked).
        pub fn __expand_try_cast_unchecked_method<T>(
            self,
            scope: &mut Scope,
        ) -> ExpandElementTyped<SliceMut<T>>
        where
            E: CubePrimitive,
            T: CubePrimitive,
        {
            if T::as_elem(scope) != E::as_elem(scope) && !is_tf32::<E, T>(scope) {
                panic!("Try cast unchecked should only be used to satisfy the rust type system.")
            }

            self.expand.into()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<SliceMut<Line<E>>> {
        /// Expand method of [with_line_size](SliceMut::with_line_size).
        pub fn __expand_with_line_size_method(
            self,
            scope: &mut Scope,
            line_size: u32,
        ) -> ExpandElementTyped<SliceMut<Line<E>>>
        where
            E: CubePrimitive,
        {
            let input = self.clone().into_variable();
            let mut item = input.item;

            if line_size as u8 == item.vectorization.unwrap_or(NonZero::new(1).unwrap()).get() {
                return self;
            }

            item.vectorization = NonZero::new(line_size as u8);
            let out = scope.create_slice(item);

            scope.register(Instruction::new(
                Operator::ReinterpretSlice(cubecl_ir::ReinterpretSliceOperator {
                    input,
                    line_size,
                }),
                *out,
            ));
            out.into()
        }
    }
}

pub(crate) fn is_tf32<C: CubePrimitive, T: CubePrimitive>(scope: &mut Scope) -> bool {
    let ty_c = C::as_elem(scope);
    let ty_t = T::as_elem(scope);
    let ty_f32 = f32::as_elem(scope);
    let ty_tf32 = tf32::as_elem(scope);

    (ty_c == ty_f32 && ty_t == ty_tf32) || (ty_c == ty_tf32 && ty_t == ty_f32)
}

impl<E: CubePrimitive> SliceOperator<E> for SharedMemory<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<SharedMemory<E>> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadOnly> {
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            start,
            end,
        )
    }

    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadWrite> {
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            start,
            end,
        )
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadOnly> {
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            0u32.into(),
            100000u32.into(), // TODO: Fix,
        )
    }

    fn __expand_to_slice_mut_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadWrite> {
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            0u32.into(),
            100000u32.into(), // TODO: Fix,
        )
    }
}

impl<E: CubePrimitive> SliceOperator<E> for Tensor<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<Tensor<E>> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadOnly> {
        SliceV2::__expand_new(scope, SliceOriginExpand::Tensor(self.clone()), start, end)
    }

    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadWrite> {
        SliceV2::__expand_new(scope, SliceOriginExpand::Tensor(self.clone()), start, end)
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::Tensor(self.clone()),
            0u32.into(),
            len,
        )
    }

    fn __expand_to_slice_mut_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::Tensor(self.clone()),
            0u32.into(),
            len,
        )
    }
}

impl<E: CubePrimitive> SliceOperator<E> for Array<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<Array<E>> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadOnly> {
        SliceV2::__expand_new(scope, SliceOriginExpand::Array(self.clone()), start, end)
    }

    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadWrite> {
        SliceV2::__expand_new(scope, SliceOriginExpand::Array(self.clone()), start, end)
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::Array(self.clone()),
            0u32.into(),
            len,
        )
    }

    fn __expand_to_slice_mut_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        SliceV2::__expand_new(
            scope,
            SliceOriginExpand::Array(self.clone()),
            0u32.into(),
            len,
        )
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceOperator<E> for SliceV2<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> SliceOperatorExpand<E> for SliceV2Expand<E, IO> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadOnly> {
        let length = crate::frontend::sub::expand(scope, end.into(), start.clone().into());
        let offset = crate::frontend::add::expand(scope, start.into(), self.offset.clone());

        SliceV2Expand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset,
            length,
        }
    }

    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadWrite> {
        let length = crate::frontend::sub::expand(scope, end.into(), start.clone().into());
        let offset = crate::frontend::add::expand(scope, start.into(), self.offset.clone());

        SliceV2Expand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset,
            length,
        }
    }

    fn __expand_to_slice_method(&self, _scope: &mut Scope) -> SliceV2Expand<E, ReadOnly> {
        SliceV2Expand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset: self.offset.clone(),
            length: self.length.clone(),
        }
    }

    fn __expand_to_slice_mut_method(&self, _scope: &mut Scope) -> SliceV2Expand<E, ReadWrite> {
        SliceV2Expand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset: self.offset.clone(),
            length: self.length.clone(),
        }
    }
}

pub trait SliceOperator<E: CubePrimitive>: CubeType<ExpandType: SliceOperatorExpand<E>> {
    /// Return a read-only view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice<Start: Index, End: Index>(&self, start: Start, end: End) -> Slice<E, ReadOnly> {
        unexpanded!()
    }
    /// Expand function of [SliceOperator::slice].
    fn __expand_slice(
        scope: &mut Scope,
        expand: Self::ExpandType,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadOnly> {
        expand.__expand_slice_method(scope, start, end)
    }

    /// Return a read-write view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice_mut<Start: Index, End: Index>(
        &mut self,
        start: Start,
        end: End,
    ) -> Slice<E, ReadWrite> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::slice_mut].
    fn __expand_slice_mut(
        scope: &mut Scope,
        expand: Self::ExpandType,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadWrite> {
        expand.__expand_slice_mut_method(scope, start, end)
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn to_slice(&self) -> Slice<E, ReadOnly> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::to_slice].
    fn __expand_to_slice(
        scope: &mut Scope,
        expand: Self::ExpandType,
    ) -> SliceV2Expand<E, ReadOnly> {
        expand.__expand_to_slice_method(scope)
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables, clippy::wrong_self_convention)]
    fn to_slice_mut(&mut self) -> SliceMut<E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::to_slice_mut].
    fn __expand_to_slice_mut(
        scope: &mut Scope,
        expand: Self::ExpandType,
    ) -> SliceV2Expand<E, ReadWrite> {
        expand.__expand_to_slice_mut_method(scope)
    }
}

pub trait SliceOperatorExpand<E: CubePrimitive>: Clone + Init + CubeDebug {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadOnly>;

    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, ReadWrite>;

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceV2Expand<E, ReadOnly>;
    fn __expand_to_slice_mut_method(&self, _scope: &mut Scope) -> SliceV2Expand<E, ReadWrite>;
}
