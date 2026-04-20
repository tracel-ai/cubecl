use super::{ReadOnly, ReadWrite, Slice, SliceExpand, SliceOriginExpand, SliceVisibility};
use crate as cubecl;
use crate::{ir::Scope, prelude::*, unexpanded};
use cubecl_common::tf32;

pub(crate) fn is_tf32<C: CubePrimitive, T: CubePrimitive>(scope: &Scope) -> bool {
    let ty_c = C::as_type(scope).storage_type();
    let ty_t = T::as_type(scope).storage_type();
    let ty_f32 = f32::as_type(scope).storage_type();
    let ty_tf32 = tf32::as_type(scope).storage_type();

    (ty_c == ty_f32 && ty_t == ty_tf32) || (ty_c == ty_tf32 && ty_t == ty_f32)
}

impl<'a, E: CubePrimitive> SliceOperator<'a, E> for SharedMemory<E> {}
impl<'a, E: CubePrimitive> SliceOperatorExpand<'a, E> for NativeExpand<SharedMemory<E>> {
    fn __expand_slice_method(
        &'a self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a SliceExpand<E, ReadOnly> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::SharedMemory(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&'a self, scope: &Scope) -> &'a SliceExpand<E, ReadOnly> {
        let len = expand_length_native(scope, self.expand);
        let slice = Slice::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(*self),
            0usize.into(),
            len.into(),
        );
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive> SliceMutOperator<'a, E> for SharedMemory<E> {}
impl<'a, E: CubePrimitive> SliceMutOperatorExpand<'a, E> for NativeExpand<SharedMemory<E>> {
    fn __expand_slice_mut_method(
        &'a mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::SharedMemory(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(
        &'a mut self,
        scope: &Scope,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let len = expand_length_native(scope, self.expand);
        let slice = Slice::__expand_new(
            scope,
            SliceOriginExpand::SharedMemory(*self),
            0usize.into(),
            len.into(),
        );
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive> SliceOperator<'a, E> for Tensor<E> {}
impl<'a, E: CubePrimitive> SliceOperatorExpand<'a, E> for NativeExpand<Tensor<E>> {
    fn __expand_slice_method(
        &'a self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a SliceExpand<E, ReadOnly> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&'a self, scope: &Scope) -> &'a SliceExpand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        let slice =
            Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive> SliceMutOperator<'a, E> for Tensor<E> {}
impl<'a, E: CubePrimitive> SliceMutOperatorExpand<'a, E> for NativeExpand<Tensor<E>> {
    fn __expand_slice_mut_method(
        &'a mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(
        &'a mut self,
        scope: &Scope,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        let slice =
            Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive> SliceOperator<'a, E> for Array<E> {}
impl<'a, E: CubePrimitive> SliceOperatorExpand<'a, E> for NativeExpand<Array<E>> {
    fn __expand_slice_method(
        &'a self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a SliceExpand<E, ReadOnly> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&'a self, scope: &Scope) -> &'a SliceExpand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive> SliceMutOperator<'a, E> for Array<E> {}
impl<'a, E: CubePrimitive> SliceMutOperatorExpand<'a, E> for NativeExpand<Array<E>> {
    fn __expand_slice_mut_method(
        &'a mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(
        &'a mut self,
        scope: &Scope,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive, IO: SliceVisibility> SliceOperator<'a, E> for Slice<E, IO> {}
impl<'a, E: CubePrimitive, IO: SliceVisibility> SliceOperatorExpand<'a, E> for SliceExpand<E, IO> {
    fn __expand_slice_method(
        &'a self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a SliceExpand<E, ReadOnly> {
        let length = end.__expand_sub_method(scope, start);
        let offset = start.__expand_add_method(scope, self.offset);
        let slice = SliceExpand {
            origin: self.origin.clone_unchecked(),
            io: core::marker::PhantomData,
            offset,
            length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&'a self, scope: &Scope) -> &'a SliceExpand<E, ReadOnly> {
        let slice = SliceExpand {
            origin: self.origin,
            io: core::marker::PhantomData,
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }
}

impl<'a, E: CubePrimitive> SliceMutOperator<'a, E> for Slice<E, ReadWrite> {}
impl<'a, E: CubePrimitive> SliceMutOperatorExpand<'a, E> for SliceExpand<E, ReadWrite> {
    fn __expand_slice_mut_method(
        &'a mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let length = end.__expand_sub_method(scope, start);
        let offset = start.__expand_add_method(scope, self.offset);

        let slice = SliceExpand {
            origin: self.origin,
            io: core::marker::PhantomData,
            offset,
            length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(
        &'a mut self,
        scope: &Scope,
    ) -> &'a mut SliceExpand<E, ReadWrite> {
        let slice = SliceExpand {
            origin: self.origin,
            io: core::marker::PhantomData,
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }
}

#[cube]
pub trait SliceOperator<'a, E: CubePrimitive> {
    /// Return a read-only view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice(&'a self, start: usize, end: usize) -> &'a Slice<E, ReadOnly> {
        unexpanded!()
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn to_slice(&'a self) -> &'a Slice<E, ReadOnly> {
        unexpanded!()
    }
}

#[cube]
pub trait SliceMutOperator<'a, E: CubePrimitive> {
    /// Return a read-write view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice_mut(&'a mut self, start: usize, end: usize) -> &'a mut Slice<E, ReadWrite> {
        unexpanded!()
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables)]
    fn to_slice_mut(&'a mut self) -> &'a mut Slice<E, ReadWrite> {
        unexpanded!()
    }
}

// Automatic implementation for references to SliceOperator.
impl<'a, T: CubePrimitive, L: SliceOperator<'a, T>> SliceOperator<'a, T> for &'a L where
    &'a L: CubeType<ExpandType = L::ExpandType>
{
}

// Automatic implementation for mutable references to SliceOperator.
impl<'a, T: CubePrimitive, L: SliceOperator<'a, T>> SliceOperator<'a, T> for &'a mut L where
    &'a mut L: CubeType<ExpandType = L::ExpandType>
{
}

// Automatic implementation for references to SliceMutOperator.
impl<'a, T: CubePrimitive, L: SliceMutOperator<'a, T>> SliceMutOperator<'a, T> for &'a L where
    &'a L: CubeType<ExpandType = L::ExpandType>
{
}

// Automatic implementation for mutable references to SliceMutOperator.
impl<'a, T: CubePrimitive, L: SliceMutOperator<'a, T>> SliceMutOperator<'a, T> for &'a mut L where
    &'a mut L: CubeType<ExpandType = L::ExpandType>
{
}
