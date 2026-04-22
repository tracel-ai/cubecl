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

impl<E: CubePrimitive> SliceOperator<E> for SharedMemory<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for NativeExpand<SharedMemory<E>> {
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E, ReadOnly> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::SharedMemory(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&self, scope: &Scope) -> &SliceExpand<E, ReadOnly> {
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

impl<E: CubePrimitive> SliceMutOperator<E> for SharedMemory<E> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for NativeExpand<SharedMemory<E>> {
    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E, ReadWrite> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::SharedMemory(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(&mut self, scope: &Scope) -> &mut SliceExpand<E, ReadWrite> {
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

impl<E: CubePrimitive> SliceOperator<E> for Tensor<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for NativeExpand<Tensor<E>> {
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E, ReadOnly> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&self, scope: &Scope) -> &SliceExpand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        let slice =
            Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<E: CubePrimitive> SliceMutOperator<E> for Tensor<E> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for NativeExpand<Tensor<E>> {
    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E, ReadWrite> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(&mut self, scope: &Scope) -> &mut SliceExpand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        let slice =
            Slice::__expand_new(scope, SliceOriginExpand::Tensor(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<E: CubePrimitive> SliceOperator<E> for Array<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for NativeExpand<Array<E>> {
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E, ReadOnly> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_method(&self, scope: &Scope) -> &SliceExpand<E, ReadOnly> {
        let len = self.clone().__expand_len_method(scope);
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<E: CubePrimitive> SliceMutOperator<E> for Array<E> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for NativeExpand<Array<E>> {
    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E, ReadWrite> {
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_to_slice_mut_method(&mut self, scope: &Scope) -> &mut SliceExpand<E, ReadWrite> {
        let len = self.clone().__expand_len_method(scope);
        let slice = Slice::__expand_new(scope, SliceOriginExpand::Array(*self), 0usize.into(), len);
        scope.create_kernel_ref(slice)
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceOperator<E> for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> SliceOperatorExpand<E> for SliceExpand<E, IO> {
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E, ReadOnly> {
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

    fn __expand_to_slice_method(&self, scope: &Scope) -> &SliceExpand<E, ReadOnly> {
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

impl<E: CubePrimitive> SliceMutOperator<E> for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> SliceMutOperatorExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E, ReadWrite> {
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

    fn __expand_to_slice_mut_method(&mut self, scope: &Scope) -> &mut SliceExpand<E, ReadWrite> {
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
pub trait SliceOperator<E: CubePrimitive> {
    /// Return a read-only view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice(&self, start: usize, end: usize) -> &Slice<E, ReadOnly> {
        unexpanded!()
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn to_slice(&self) -> &Slice<E, ReadOnly> {
        unexpanded!()
    }
}

#[cube]
pub trait SliceMutOperator<E: CubePrimitive> {
    /// Return a read-write view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice_mut(&mut self, start: usize, end: usize) -> &mut Slice<E, ReadWrite> {
        unexpanded!()
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables)]
    fn to_slice_mut(&mut self) -> &mut Slice<E, ReadWrite> {
        unexpanded!()
    }
}
