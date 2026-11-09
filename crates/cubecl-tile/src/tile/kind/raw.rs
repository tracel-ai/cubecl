//! Values a tile holds apart from the element their type names, each named again where it is
//! used: a value of a type naming one element ([`Raw`]: a slice, registers, a tensor map) and an
//! erased call ([`RawCall`]). A cmma fragment and a warpgroup accumulator need neither: they are
//! kept at [`Held`] and named again with their own `downcast`.
use std::marker::PhantomData;

use cubecl::define_size;
use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ElemType, ExpandValue, Scope, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedIo, ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, ReadsLines,
    WriteOnly, WritesLines,
};

use crate::Held;

/// Which type a [`Raw`] value is, at whatever element it holds.
pub(crate) trait RawKind: 'static {
    /// The type at element `T`.
    type At<T: Numeric>: CubeType<ExpandType = NativeExpand<Self::At<T>>>;
}

/// A slice of a buffer.
pub(crate) struct Slice;
/// A slice of shared memory.
pub(crate) struct SharedSlice;
/// A unit's registers, in lines of `W`.
pub(crate) struct Registers<W: Size>(PhantomData<W>);
/// A TMA tensor map of tiled boxes.
pub(crate) struct TiledMap;

impl RawKind for Slice {
    type At<T: Numeric> = Box<[T]>;
}

impl RawKind for SharedSlice {
    type At<T: Numeric> = Shared<[T]>;
}

impl<W: Size> RawKind for Registers<W> {
    type At<T: Numeric> = Array<Vector<T, W>>;
}

impl RawKind for TiledMap {
    type At<T: Numeric> = TensorMap<T, Tiled>;
}

/// A value of the type `K` names, held as the value and the element it holds.
pub(crate) struct Raw<K: RawKind> {
    _kind: PhantomData<K>,
}

pub(crate) struct RawExpand<K: RawKind> {
    value: ExpandValue,
    held: ElemType,
    _kind: PhantomData<K>,
}

impl<K: RawKind> Raw<K> {
    /// `value`, its element kept beside it.
    pub(crate) fn of<T: Numeric>(_value: K::At<T>) -> Raw<K> {
        unexpanded!()
    }

    /// The value, named at `T`, the element it holds.
    pub(crate) fn at<T: Numeric>(&self) -> &K::At<T> {
        unexpanded!()
    }

    /// The mutable twin of [`at`](Self::at).
    pub(crate) fn at_mut<T: Numeric>(&mut self) -> &mut K::At<T> {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        scope: &Scope,
        value: NativeExpand<K::At<T>>,
    ) -> RawExpand<K> {
        RawExpand {
            value: value.expand,
            held: T::elem_type(scope),
            _kind: PhantomData,
        }
    }
}

impl<K: RawKind> RawExpand<K> {
    pub(crate) fn __expand_at_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a NativeExpand<K::At<T>> {
        scope.create_kernel_ref(self.typed::<T>(scope))
    }

    pub(crate) fn __expand_at_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut NativeExpand<K::At<T>> {
        scope.create_kernel_ref(self.typed::<T>(scope))
    }

    /// The value named at `T`, which must be the element it holds.
    fn typed<T: Numeric>(&self, scope: &Scope) -> NativeExpand<K::At<T>> {
        let named = T::elem_type(scope);
        assert_eq!(
            self.held, named,
            "Raw: the value holds {:?}, not {named:?}",
            self.held
        );
        self.value.into()
    }
}

impl Raw<Slice> {
    /// Free the shared memory this slice is a window of, which the allocation traces back to its
    /// declaration.
    pub(crate) fn free(&self) {
        unexpanded!()
    }
}

impl RawExpand<Slice> {
    pub(crate) fn __expand_free_method(&self, scope: &Scope) {
        // The list a slice is a window of is the same whatever element the slice is named at.
        let slice: NativeExpand<Box<[u32]>> = self.value.into();
        let list = slice.__extract_list(scope);
        scope.register(&cubecl::ir::dialect::general::FreeOp::new(
            scope.ctx_mut(),
            list,
        ));
    }
}

/// An erased call, read from or written to, held apart from its element.
pub(crate) struct RawCall<IO> {
    _io: PhantomData<IO>,
}

pub(crate) struct RawCallExpand<IO> {
    call: ErasedTensorExpand<Held, IO>,
    held: ElemType,
}

impl<IO> Clone for RawCall<IO> {
    fn clone(&self) -> Self {
        RawCall { _io: PhantomData }
    }
}

impl<IO> Clone for RawCallExpand<IO> {
    fn clone(&self) -> Self {
        RawCallExpand {
            call: self.call.clone(),
            held: self.held,
        }
    }
}

impl<IO: RenamedIo> RawCall<IO> {
    /// `call`, its element kept beside it.
    pub(crate) fn of<T: Numeric>(_call: ErasedTensor<T, IO>) -> RawCall<IO> {
        unexpanded!()
    }

    /// The call, named at `T`, the element it holds.
    pub(crate) fn call_mut<T: Numeric>(&mut self) -> &mut ErasedTensor<T, IO> {
        unexpanded!()
    }

    /// The read-only twin of [`call_mut`](Self::call_mut).
    pub(crate) fn call<T: Numeric>(&self) -> &ErasedTensor<T, IO> {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        scope: &Scope,
        call: ErasedTensorExpand<T, IO>,
    ) -> RawCallExpand<IO> {
        RawCallExpand {
            call: IO::renamed(call),
            held: T::elem_type(scope),
        }
    }
}

impl<IO: RenamedIo> RawCallExpand<IO> {
    pub(crate) fn __expand_call_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut ErasedTensorExpand<T, IO> {
        scope.create_kernel_ref(self.named::<T>(scope))
    }

    pub(crate) fn __expand_call_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a ErasedTensorExpand<T, IO> {
        scope.create_kernel_ref(self.named::<T>(scope))
    }

    /// The call named at `T`, which must be the element it holds.
    fn named<T: Numeric>(&self, scope: &Scope) -> ErasedTensorExpand<T, IO> {
        let named = T::elem_type(scope);
        assert_eq!(
            self.held, named,
            "RawCall: the call holds {:?}, not {named:?}",
            self.held
        );
        IO::renamed(self.call.clone())
    }
}

/// The visibilities a [`RawCall`] is read or written at, each renaming a call through the half
/// of the interface it offers.
pub(crate) trait RenamedIo: ErasedIo {
    /// `call`'s lines, served at `O`: the lines cross untyped, so only the name changes.
    fn renamed<E: Numeric, O: Numeric>(
        call: ErasedTensorExpand<E, Self>,
    ) -> ErasedTensorExpand<O, Self>;
}

impl RenamedIo for ReadOnly {
    fn renamed<E: Numeric, O: Numeric>(
        call: ErasedTensorExpand<E, Self>,
    ) -> ErasedTensorExpand<O, Self> {
        ErasedTensorExpand::new(Renamed { call })
    }
}

impl RenamedIo for WriteOnly {
    fn renamed<E: Numeric, O: Numeric>(
        call: ErasedTensorExpand<E, Self>,
    ) -> ErasedTensorExpand<O, Self> {
        ErasedTensorExpand::new(Renamed { call })
    }
}

/// A call's backing at another element than its own.
struct Renamed<E: Numeric, IO> {
    call: ErasedTensorExpand<E, IO>,
}

// The width a renamed line crosses at, bound to the call's own before each access.
define_size!(Width);

impl<E: Numeric, O: Numeric> ErasedTensorOperationsExpand<O> for Renamed<E, ReadOnly> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.call.__expand_vector_size_method(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.call.__expand_len_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        scope.register_size::<Width>(self.call.__expand_vector_size_method(scope));
        self.call.__expand_read_method::<Width>(scope, index).expand
    }
}

impl<E: Numeric, O: Numeric> ReadsLines<O> for Renamed<E, ReadOnly> {}

impl<E: Numeric, O: Numeric> ErasedTensorOperationsExpand<O> for Renamed<E, WriteOnly> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.call.__expand_vector_size_method(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.call.__expand_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        scope.register_size::<Width>(self.call.__expand_vector_size_method(scope));
        self.call
            .__expand_write_method::<Width>(scope, index, value.into())
    }
}

impl<E: Numeric, O: Numeric> WritesLines<O> for Renamed<E, WriteOnly> {}

impl<K: RawKind> Clone for Raw<K> {
    fn clone(&self) -> Self {
        Raw { _kind: PhantomData }
    }
}

impl<K: RawKind> Clone for RawExpand<K> {
    fn clone(&self) -> Self {
        RawExpand {
            value: self.value,
            held: self.held,
            _kind: PhantomData,
        }
    }
}

impl<K: RawKind> CubeType for Raw<K> {
    type ExpandType = RawExpand<K>;
}

impl<IO: RenamedIo> CubeType for RawCall<IO> {
    type ExpandType = RawCallExpand<IO>;
}

macro_rules! expand_value {
    ($ty: ty $(, $param: ident: $bound: tt)?) => {
        impl$(<$param: $bound>)? IntoExpand for $ty {
            type Expand = Self;
            fn into_expand(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$param: $bound>)? ExpandTypeClone for $ty {
            fn clone_unchecked(&self) -> Self {
                self.clone()
            }
        }
        impl$(<$param: $bound>)? IntoMut for $ty {
            fn into_mut(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$param: $bound>)? CubeDebug for $ty {}
        impl$(<$param: $bound>)? AsRefExpand for $ty {
            fn __expand_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl$(<$param: $bound>)? AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }
    };
}

expand_value!(RawExpand<K>, K: RawKind);
expand_value!(RawCallExpand<IO>, IO: 'static);

#[cfg(test)]
mod tests {
    use cubecl::ir::{AddressType, settings};
    use half::f16;

    use super::*;

    fn test_scope() -> Scope {
        Scope::root(settings::KernelSettings::new(
            settings::Dim3::new_single(),
            settings::ExecutionMode::Checked,
            AddressType::U32,
        ))
    }

    /// A slice of shared memory held at `H`, then named at `N`.
    #[cube]
    fn slice_named_at<H: Numeric, N: Numeric>() {
        let smem = Shared::<[H]>::new_slice(4usize);
        let raw = Raw::<Slice>::of::<H>(unsafe { smem.inner_ref().as_boxed_unchecked() });
        let _ = raw.at::<N>();
    }

    /// A slice is named at the element it holds.
    #[test]
    fn a_slice_is_named_at_its_element() {
        slice_named_at::expand::<f16, f16>(&test_scope());
    }

    /// A slice named at an element it does not hold is refused at expansion.
    ///
    /// Read at the wrong element, a slice addresses the wrong bytes: an `f16` slice read as `f32`
    /// takes two values for one, and nothing at runtime notices.
    #[test]
    #[should_panic(expected = "Raw: the value holds")]
    fn a_slice_refuses_another_element() {
        slice_named_at::expand::<f16, f32>(&test_scope());
    }

    /// A backing of four one-wide lines that the tests name but never read or write.
    struct Unread;

    impl<E: Numeric> ErasedTensorOperationsExpand<E> for Unread {
        fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
            1
        }

        fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
            4usize.into_expand(scope)
        }
    }

    impl<E: Numeric> ReadsLines<E> for Unread {}
    impl<E: Numeric> WritesLines<E> for Unread {}

    /// An erased call is named at the element it holds, read or written.
    #[test]
    fn an_erased_call_is_named_at_its_element() {
        let scope = test_scope();
        let read = RawCall::<ReadOnly>::__expand_of::<f16>(&scope, ErasedTensorExpand::new(Unread));
        let _ = read.__expand_call_method::<f16>(&scope);
        let mut write =
            RawCall::<WriteOnly>::__expand_of::<f16>(&scope, ErasedTensorExpand::new(Unread));
        let _ = write.__expand_call_mut_method::<f16>(&scope);
    }

    /// An erased call named at an element it does not hold is refused at expansion.
    ///
    /// The call's lines cross untyped, so a call named at the wrong element would hand its
    /// backing values of another width than it serves.
    #[test]
    #[should_panic(expected = "RawCall: the call holds")]
    fn an_erased_call_refuses_another_element() {
        let scope = test_scope();
        let read = RawCall::<ReadOnly>::__expand_of::<f16>(&scope, ErasedTensorExpand::new(Unread));
        let _ = read.__expand_call_method::<f32>(&scope);
    }
}
