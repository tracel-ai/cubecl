//! A write destination whose line width is not part of its type.
//!
//! [`VirtualTensor`] already decouples a tensor from how it is stored, but it
//! carries the line width as a type parameter (`Vector<E, N>`). That is right
//! for a kernel whose operand type is written down, and wrong for an engine that
//! erases the width on purpose so one tile type serves every operand — such an
//! engine cannot name `N` in the field that would hold the destination.
//!
//! [`ErasedSink`] is the same decoupling with the width moved from the type to
//! the call. It is built where the width *is* known (expansion, from a
//! [`VirtualTensor`] or anything else that can take a line) and thereafter
//! travels width-free; each write states the width it is writing at, and the
//! sink checks that it is the one it was built for.
//!
//! # Write-only, deliberately
//!
//! A sink is not a tensor with the reads left out. It exists for destinations
//! that are *not memory* — a value handed to a generated epilogue rather than
//! stored — and those have no address to read back from. A destination that can
//! be read is a [`VirtualTensor`] and should be one.

use alloc::sync::Arc;
use core::{cell::UnsafeCell, marker::PhantomData};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    ir::{ExpandValue, VectorSize},
    unexpanded,
};

use crate::tensor::r#virtual::{VirtualTensor, VirtualTensorExpand};
use crate::tensor::{
    ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand,
    layout::Coords1d,
};
use cubecl_core::prelude::barrier::Barrier;

/// A write destination that has forgotten its line width. See the
/// [module docs](self).
#[derive(Clone)]
pub struct ErasedSink<E: Numeric> {
    _e: PhantomData<E>,
}

/// Expand type for [`ErasedSink`].
#[derive(Clone)]
pub struct ErasedSinkExpand<E: Numeric> {
    state: Arc<UnsafeCell<dyn ErasedSinkOperationsExpand<E>>>,
}

/// What a destination has to answer to be an [`ErasedSink`].
///
/// Expand-only, and that is the whole point: the value crosses as an
/// [`ExpandValue`], which is untyped in the IR, so the trait can be object-safe
/// without the width appearing in it. The implementor knows its own width and is
/// the one that can check the caller's against it.
pub trait ErasedSinkOperationsExpand<E: Numeric> {
    /// The line width this destination takes, in scalars.
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize;

    /// Write one line at `index`, counted in lines of
    /// [`vector_size`](Self::__expand_vector_size_method).
    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    );

    /// How many lines the destination takes, which is what a *checked* write is
    /// checked against. A destination with no bound of its own answers with one
    /// no index reaches.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize>;
}

impl<E: Numeric> ErasedSinkExpand<E> {
    /// Erase `sink`'s width.
    ///
    /// Takes the *expand*, because that is where a width is known: the caller
    /// has the `N`-typed destination in hand and gives up naming it here.
    pub fn new<S: ErasedSinkOperationsExpand<E> + 'static>(sink: S) -> Self {
        Self {
            state: Arc::new(UnsafeCell::new(sink)),
        }
    }

    fn state_read(&self) -> &dyn ErasedSinkOperationsExpand<E> {
        // SAFETY: the state is valid for the whole lifetime of `self`.
        unsafe { &mut *self.state.get() }
    }

    #[allow(clippy::mut_from_ref)]
    fn state_write(&self) -> &mut dyn ErasedSinkOperationsExpand<E> {
        // SAFETY: as `VirtualTensorExpand`: the state is a handle into memory
        // and only the memory is written; the state itself is never mutated.
        unsafe { &mut *self.state.get() }
    }
}

impl<E: Numeric, N: Size> From<VirtualTensorExpand<E, N, ReadWrite>> for ErasedSinkExpand<E> {
    fn from(tensor: VirtualTensorExpand<E, N, ReadWrite>) -> Self {
        ErasedSinkExpand::new(tensor)
    }
}

#[cube]
impl<E: Numeric> ErasedSink<E> {
    /// Write one `N`-wide line at `index`, counted in lines.
    ///
    /// # Panics
    ///
    /// At expansion, when `N` is not the width the sink was built for. The
    /// alternative is a store that lands on a different element than the caller
    /// named, which no runtime check would catch.
    #[allow(unused)]
    pub fn write<N: Size>(&mut self, index: usize, value: Vector<E, N>) {
        intrinsic!(|scope| {
            let served = self.state_read().__expand_vector_size_method(scope);
            let asked = <N as Size>::__expand_value(scope);
            assert_eq!(
                served, asked,
                "ErasedSink::write: the sink takes {served}-wide lines and the write is {asked}-wide"
            );
            self.state_write()
                .__expand_write_line_method(scope, index, value.into())
        })
    }
}

impl<E: Numeric> Vectorized for ErasedSink<E> {}
impl<E: Numeric> VectorizedExpand for ErasedSinkExpand<E> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.state_read().__expand_vector_size_method(scope)
    }
}

/// A [`VirtualTensor`] is the destination an erased sink most often wraps, so it
/// is one without the caller writing an adapter.
impl<E: Numeric, N: Size> ErasedSinkOperationsExpand<E> for VirtualTensorExpand<E, N, ReadWrite> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        VectorizedExpand::__expand_vector_size_method(self, scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        self.state_write()
            .__expand_write_method(scope, index, value.into())
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.clone().__expand_len_method(scope)
    }
}

/// A launched tensor as a sink.
///
/// The degenerate case — a destination that *is* memory, reached through the
/// indirection anyway — and the one a test compares the interesting cases
/// against.
impl<E: Numeric, N: Size> ErasedSinkOperationsExpand<E> for TensorExpand<Vector<E, N>> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        VectorizedExpand::__expand_vector_size_method(self, scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        unsafe {
            self.__expand_get_unchecked_mut_method(scope, index)
                .__expand_assign_method(scope, value.into())
        };
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }
}

/// Any mutable view, at a width named once here and forgotten after.
///
/// The general way in, and the reason the other two are conveniences rather than
/// the interface: a destination that can already take a line at `Coords1d` is a
/// sink, whatever it does with it. What this adds is only the erasure — `N` is
/// captured at construction and reappears when the write is made.
pub struct SinkOfView<V, N: Size> {
    view: V,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size, V> ErasedSinkOperationsExpand<E> for SinkOfView<V, N>
where
    V: ViewOperationsMutExpand<Vector<E, N>, Coords1d>,
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        self.view.__expand_write_method(scope, index, value.into())
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        <V as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_shape_method(
            &self.view, scope,
        )
    }
}

impl<E: Numeric> ErasedSink<E> {
    /// The sink that writes into `view`, whose lines are `N` wide.
    ///
    /// The width is named at this call and nowhere after it, which is the whole
    /// point of the type: a caller that knows `N` hands it over here so the
    /// engine holding the sink never has to.
    pub fn of_view<V: CubeType, N: Size>(_view: V) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_view`](Self::of_view).
    pub fn __expand_of_view<V: CubeType, N: Size>(
        _scope: &Scope,
        view: V::ExpandType,
    ) -> ErasedSinkExpand<E>
    where
        V::ExpandType: ViewOperationsMutExpand<Vector<E, N>, Coords1d> + 'static,
    {
        ErasedSinkExpand::new(SinkOfView::<V::ExpandType, N> {
            view,
            _n: PhantomData,
        })
    }

    /// The sink that writes into `tensor` — memory, through the indirection.
    pub fn of_tensor<N: Size>(_tensor: &Tensor<Vector<E, N>>) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_tensor`](Self::of_tensor).
    pub fn __expand_of_tensor<N: Size>(
        _scope: &Scope,
        tensor: &TensorExpand<Vector<E, N>>,
    ) -> ErasedSinkExpand<E> {
        ErasedSinkExpand::new(ExpandTypeClone::clone_unchecked(tensor))
    }

    /// The sink that writes into `tensor`.
    pub fn of_virtual<N: Size>(_tensor: VirtualTensor<E, N, ReadWrite>) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_virtual`](Self::of_virtual).
    pub fn __expand_of_virtual<N: Size>(
        _scope: &Scope,
        tensor: VirtualTensorExpand<E, N, ReadWrite>,
    ) -> ErasedSinkExpand<E> {
        ErasedSinkExpand::new(tensor)
    }
}

/// Making [`ErasedSink`] a proper [cube type](CubeType), the same way
/// [`VirtualTensor`] is one.
mod __cube_type {
    use super::*;

    impl<E: Numeric> CubeType for ErasedSink<E> {
        type ExpandType = ErasedSinkExpand<E>;
    }

    impl<E: Numeric> IntoExpand for ErasedSinkExpand<E> {
        type Expand = ErasedSinkExpand<E>;

        fn into_expand(self, _: &Scope) -> Self::Expand {
            self
        }
    }

    impl<E: Numeric> ExpandTypeClone for ErasedSinkExpand<E> {
        fn clone_unchecked(&self) -> Self {
            self.clone()
        }
    }

    impl<E: Numeric> IntoMut for ErasedSinkExpand<E> {
        fn into_mut(self, _scope: &Scope) -> Self {
            self
        }
    }

    impl<E: Numeric> CubeDebug for ErasedSinkExpand<E> {}

    impl<E: Numeric> AsRefExpand for ErasedSinkExpand<E> {
        fn __expand_ref_method(&self, _: &Scope) -> &Self {
            self
        }
    }

    impl<E: Numeric> AsMutExpand for ErasedSinkExpand<E> {
        fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
            self
        }
    }
}

/// The sink as the backing of a [`ViewMut`], which is how a kernel that
/// addresses its destination through a layout reaches one.
///
/// `N` is constrained by the trait rather than by the type, which is the whole
/// trick: the sink stays width-free, and the width arrives with the view the
/// caller builds over it. [`write`](ErasedSink::write) checks the two agree.
///
/// The read half is present because [`ViewOperationsMut`] requires it and
/// absent in every other sense: see the [module docs](self) for why a sink has
/// nothing to read back.
impl<E: Numeric, N: Size> ViewOperations<Vector<E, N>, Coords1d> for ErasedSink<E> {}

impl<E: Numeric, N: Size> ViewOperationsExpand<Vector<E, N>, Coords1d> for ErasedSinkExpand<E> {
    fn __expand_read_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        unimplemented!("ErasedSink: a sink is written, never read")
    }

    fn __expand_read_checked_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        unimplemented!("ErasedSink: a sink is written, never read")
    }

    fn __expand_read_masked_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _mask_value: <Vector<E, N> as CubeType>::ExpandType,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        unimplemented!("ErasedSink: a sink is written, never read")
    }

    fn __expand_read_unchecked_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        unimplemented!("ErasedSink: a sink is written, never read")
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _end: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<E, N>> {
        unimplemented!("ErasedSink: a sink has no address to slice")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.state_read().__expand_lines_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<bool> {
        let lines = self.state_read().__expand_lines_method(scope);
        pos.__expand_lt_method(scope, &lines)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<E, N>>,
        _pos: NativeExpand<usize>,
    ) {
        unimplemented!("ErasedSink: not a tensor map")
    }
}

impl<E: Numeric, N: Size> ViewOperationsMut<Vector<E, N>, Coords1d> for ErasedSink<E> {}

impl<E: Numeric, N: Size> ViewOperationsMutExpand<Vector<E, N>, Coords1d> for ErasedSinkExpand<E> {
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <Vector<E, N> as CubeType>::ExpandType,
    ) {
        let served = self.state_read().__expand_vector_size_method(scope);
        let asked = <N as Size>::__expand_value(scope);
        assert_eq!(
            served, asked,
            "ErasedSink: the sink takes {served}-wide lines and the view writes {asked}-wide"
        );
        self.state_write()
            .__expand_write_line_method(scope, pos, value.into())
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <Vector<E, N> as CubeType>::ExpandType,
    ) {
        let in_bounds =
            <Self as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_is_in_bounds_method(
                self, scope, pos,
            );
        if_expand(scope, in_bounds, |scope| {
            <Self as ViewOperationsMutExpand<Vector<E, N>, Coords1d>>::__expand_write_method(
                self, scope, pos, value,
            )
        })
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _end: NativeExpand<usize>,
    ) -> &mut SliceExpand<Vector<E, N>> {
        unimplemented!("ErasedSink: a sink has no address to slice")
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &Scope,
        _shared_memory: &SliceExpand<Vector<E, N>>,
        _pos: NativeExpand<usize>,
    ) {
        unimplemented!("ErasedSink: not a tensor map")
    }
}
