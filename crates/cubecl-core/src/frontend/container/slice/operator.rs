use super::{ReadOnly, ReadWrite, Slice, SliceExpand, SliceOriginExpand, SliceVisibility};
use crate::{
    frontend::{Array, CubePrimitive, CubeType, ExpandElementTyped, indexation::Index},
    ir::Scope,
    prelude::{SharedMemory, Tensor, expand_length_native},
    unexpanded,
};
use cubecl_common::tf32;
use cubecl_ir::ExpandElement;

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
    ) -> SliceExpand<E, ReadOnly> {
        Slice::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            start,
            end,
        )
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadOnly> {
        let len = expand_length_native(scope, *self.expand);

        Slice::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            0u32.into(),
            ExpandElement::Plain(len).into(),
        )
    }
}

impl<E: CubePrimitive> SliceMutOperator<E> for SharedMemory<E> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for ExpandElementTyped<SharedMemory<E>> {
    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadWrite> {
        Slice::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            start,
            end,
        )
    }

    fn __expand_to_slice_mut_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadWrite> {
        let len = expand_length_native(scope, *self.expand);

        Slice::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(self.clone()),
            0u32.into(),
            ExpandElement::Plain(len).into(),
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
    ) -> SliceExpand<E, ReadOnly> {
        Slice::__expand_new(scope, SliceOriginExpand::Tensor(self.clone()), start, end)
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        Slice::__expand_new(
            scope,
            SliceOriginExpand::Tensor(self.clone()),
            0u32.into(),
            len,
        )
    }
}

impl<E: CubePrimitive> SliceMutOperator<E> for Tensor<E> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for ExpandElementTyped<Tensor<E>> {
    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadWrite> {
        Slice::__expand_new(scope, SliceOriginExpand::Tensor(self.clone()), start, end)
    }

    fn __expand_to_slice_mut_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        Slice::__expand_new(
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
    ) -> SliceExpand<E, ReadOnly> {
        Slice::__expand_new(scope, SliceOriginExpand::Array(self.clone()), start, end)
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        Slice::__expand_new(
            scope,
            SliceOriginExpand::Array(self.clone()),
            0u32.into(),
            len,
        )
    }
}

impl<E: CubePrimitive> SliceMutOperator<E> for Array<E> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for ExpandElementTyped<Array<E>> {
    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadWrite> {
        Slice::__expand_new(scope, SliceOriginExpand::Array(self.clone()), start, end)
    }

    fn __expand_to_slice_mut_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        Slice::__expand_new(
            scope,
            SliceOriginExpand::Array(self.clone()),
            0u32.into(),
            len,
        )
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceOperator<E> for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> SliceOperatorExpand<E> for SliceExpand<E, IO> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadOnly> {
        let length = crate::frontend::sub::expand(scope, end, start.clone());
        let offset = crate::frontend::add::expand(scope, start, self.offset.clone());

        SliceExpand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset,
            length,
            line_size: None,
        }
    }

    fn __expand_to_slice_method(&self, _scope: &mut Scope) -> SliceExpand<E, ReadOnly> {
        SliceExpand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset: self.offset.clone(),
            length: self.length.clone(),
            line_size: self.line_size,
        }
    }
}

impl<E: CubePrimitive> SliceMutOperator<E> for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadWrite> {
        let length = crate::frontend::sub::expand(scope, end, start.clone());
        let offset = crate::frontend::add::expand(scope, start, self.offset.clone());

        SliceExpand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset,
            length,
            line_size: None,
        }
    }

    fn __expand_to_slice_mut_method(&self, _scope: &mut Scope) -> SliceExpand<E, ReadWrite> {
        SliceExpand {
            origin: self.origin.clone(),
            io: std::marker::PhantomData,
            offset: self.offset.clone(),
            length: self.length.clone(),
            line_size: self.line_size,
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
    ) -> SliceExpand<E, ReadOnly> {
        expand.__expand_slice_method(scope, start, end)
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn to_slice(&self) -> Slice<E, ReadOnly> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::to_slice].
    fn __expand_to_slice(scope: &mut Scope, expand: Self::ExpandType) -> SliceExpand<E, ReadOnly> {
        expand.__expand_to_slice_method(scope)
    }
}

pub trait SliceMutOperator<E: CubePrimitive>:
    CubeType<ExpandType: SliceMutOperatorExpand<E>>
{
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

    /// Expand function of [SliceMutOperator::slice_mut].
    fn __expand_slice_mut(
        scope: &mut Scope,
        expand: Self::ExpandType,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadWrite> {
        expand.__expand_slice_mut_method(scope, start, end)
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables)]
    fn to_slice_mut(&mut self) -> Slice<E, ReadWrite> {
        unexpanded!()
    }

    /// Expand function of [SliceMutOperator::to_slice_mut].
    fn __expand_to_slice_mut(
        scope: &mut Scope,
        expand: Self::ExpandType,
    ) -> SliceExpand<E, ReadWrite> {
        expand.__expand_to_slice_mut_method(scope)
    }
}

pub trait SliceOperatorExpand<E: CubePrimitive> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadOnly>;

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceExpand<E, ReadOnly>;
}

pub trait SliceMutOperatorExpand<E: CubePrimitive> {
    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, ReadWrite>;

    fn __expand_to_slice_mut_method(&self, _scope: &mut Scope) -> SliceExpand<E, ReadWrite>;
}

// Automatic implementation for references to List.
impl<'a, T: CubePrimitive, L: SliceOperator<T>> SliceOperator<T> for &'a L where
    &'a L: CubeType<ExpandType = L::ExpandType>
{
}

// Automatic implementation for mutable references to List.
impl<'a, T: CubePrimitive, L: SliceOperator<T>> SliceOperator<T> for &'a mut L where
    &'a mut L: CubeType<ExpandType = L::ExpandType>
{
}

// Automatic implementation for references to ListMut.
impl<'a, T: CubePrimitive, L: SliceMutOperator<T>> SliceMutOperator<T> for &'a L where
    &'a L: CubeType<ExpandType = L::ExpandType>
{
}

// Automatic implementation for references to ListMut.
impl<'a, T: CubePrimitive, L: SliceMutOperator<T>> SliceMutOperator<T> for &'a mut L where
    &'a mut L: CubeType<ExpandType = L::ExpandType>
{
}
