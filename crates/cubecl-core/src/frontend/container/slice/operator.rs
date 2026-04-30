use alloc::boxed::Box;

use super::{SliceExpand, SliceOriginExpand};
use crate::{self as cubecl, frontend::ranges::range};
use crate::{ir::Scope, prelude::*, unexpanded};
use cubecl_common::tf32;

pub(crate) fn is_tf32<C: CubePrimitive, T: CubePrimitive>(scope: &Scope) -> bool {
    let ty_c = C::__expand_as_type(scope).storage_type();
    let ty_t = T::__expand_as_type(scope).storage_type();
    let ty_f32 = f32::__expand_as_type(scope).storage_type();
    let ty_tf32 = tf32::__expand_as_type(scope).storage_type();

    (ty_c == ty_f32 && ty_t == ty_tf32) || (ty_c == ty_tf32 && ty_t == ty_f32)
}

type ArrayExpand<E> = NativeExpand<Array<E>>;
type TensorExpand<E> = NativeExpand<Tensor<E>>;
type SharedMemoryExpand<E> = NativeExpand<SharedMemory<E>>;

impl<E: CubePrimitive> SliceOperator<E> for SharedMemory<E> {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for NativeExpand<SharedMemory<E>> {
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E> {
        let slice = SliceExpand::new(scope, SliceOriginExpand::SharedMemory(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E> {
        let slice = SliceExpand::new(scope, SliceOriginExpand::SharedMemory(*self), start, end);
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
    ) -> &SliceExpand<E> {
        let slice = SliceExpand::new(scope, SliceOriginExpand::Tensor(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E> {
        let slice = SliceExpand::new(scope, SliceOriginExpand::Tensor(*self), start, end);
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
    ) -> &SliceExpand<E> {
        let slice = SliceExpand::new(scope, SliceOriginExpand::Array(*self), start, end);
        scope.create_kernel_ref(slice)
    }

    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E> {
        let slice = SliceExpand::new(scope, SliceOriginExpand::Array(*self), start, end);
        scope.create_kernel_ref(slice)
    }
}

#[cube]
impl<E: CubePrimitive> Array<E> {
    pub fn as_slice(&self) -> &[E] {
        let len = self.buffer_len();
        &self[..len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [E] {
        let len = self.buffer_len();
        &mut self[..len]
    }
}

#[cube]
impl<E: CubePrimitive> Tensor<E> {
    pub fn as_slice(&self) -> &[E] {
        let len = self.buffer_len();
        &self[..len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [E] {
        let len = self.buffer_len();
        &mut self[..len]
    }
}

#[cube]
impl<E: CubePrimitive> SharedMemory<E> {
    pub fn as_slice(&self) -> &[E] {
        &self[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [E] {
        &mut self[..]
    }
}

impl<E: CubePrimitive> SliceOperator<E> for Box<[E]> {}
impl<E: CubePrimitive> SliceOperator<E> for [E] {}
impl<E: CubePrimitive> SliceOperatorExpand<E> for SliceExpand<E> {
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<E> {
        let length = end.__expand_sub_method(scope, start);
        let offset = start.__expand_add_method(scope, self.offset);
        let slice = SliceExpand {
            origin: self.origin.clone_unchecked(),
            offset,
            length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }

    fn __expand_slice_mut_method(
        &mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<E> {
        let length = end.__expand_sub_method(scope, start);
        let offset = start.__expand_add_method(scope, self.offset);

        let slice = SliceExpand {
            origin: self.origin,
            offset,
            length,
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
    fn slice(&self, start: usize, end: usize) -> &[E] {
        unexpanded!()
    }

    /// Return a read-write view of all elements comprise between the `start` and `end` indices.
    /// In `checked` mode, if the `end` index is out-of-bound, it is replaced by
    /// the length of `self`.
    #[allow(unused_variables)]
    fn slice_mut(&mut self, start: usize, end: usize) -> &mut [E] {
        unexpanded!()
    }
}

// Simple heuristic
const MEMCPY_UNROLL_LIMIT: usize = 8;

pub trait MemcpyExpand<E: CubePrimitive>: ListExpand<E> {
    fn __expand_copy_from_slice_method(&mut self, scope: &Scope, source: &SliceExpand<E>) {
        let dest = self.__expand_slice_mut_method(
            scope,
            NativeExpand::from_lit(scope, 0),
            self.__expand_len_method(scope),
        );
        let len = source.__expand_len_method(scope);
        let unroll = source
            .const_len()
            .is_some_and(|it| it <= MEMCPY_UNROLL_LIMIT);
        for_expand(
            scope,
            range::expand(scope, 0usize.into_expand(scope), len),
            unroll,
            |scope, idx| {
                copy::expand(
                    scope,
                    source.__expand_index_method(scope, idx),
                    dest.__expand_index_mut_method(scope, idx),
                );
            },
        );
    }
}

impl<E: CubePrimitive, T: ListExpand<E>> MemcpyExpand<E> for T {}
