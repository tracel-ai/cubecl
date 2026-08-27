//! A tensor whose line width is not part of its type.
//!
//! [`VirtualTensor`] already decouples a tensor from how it is stored, but it
//! carries the line width as a type parameter (`Vector<E, N>`). That is right
//! for a kernel whose operand type is written down, and wrong for an engine that
//! erases the width on purpose so one tile type serves every operand — such an
//! engine cannot name `N` in the field that would hold the tensor.
//!
//! [`ErasedTensor`] is the same decoupling with the width moved from the type to
//! the call: it is [`VirtualTensor`] with one fewer type parameter. It is built
//! where the width *is* known (expansion, from a [`VirtualTensor`], a launched
//! tensor, or any view) and thereafter travels width-free; each access states
//! the width it is working at, and the tensor checks that it is the one it was
//! built for.
//!
//! The value crosses the trait boundary as an [`ExpandValue`], which is untyped
//! in the IR, so [`ErasedTensorOperationsExpand`] is object-safe without the
//! width appearing in it at all.
//!
//! # Visibility
//!
//! Like [`VirtualTensor`], an erased tensor carries an `IO` marker saying which
//! half of the interface it offers: [`ReadOnly`], [`ReadWrite`], or
//! [`WriteOnly`]. Unlike [`VirtualTensor`], the marker is not merely decorative —
//! the constructors demand the matching capability of the backing (see
//! [`ReadsLines`] and [`WritesLines`]), so a destination that cannot read cannot
//! be wrapped as one that can.
//!
//! [`WriteOnly`] exists because the interesting destinations are *not memory* — a
//! value handed to a generated epilogue rather than stored — and those have no
//! address to read back from.

use alloc::sync::Arc;
use core::{cell::UnsafeCell, marker::PhantomData};
use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    frontend::select,
    ir::{ExpandValue, VectorSize},
    unexpanded,
};

use crate::tensor::r#virtual::{VirtualTensor, VirtualTensorExpand};
use crate::tensor::{
    ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand,
    layout::Coords1d,
};
use cubecl_core::prelude::barrier::Barrier;

/// Visibility marker for a destination that is written and never read.
///
/// Not defined next to [`ReadOnly`] and [`ReadWrite`] in `cubecl_core`, because
/// those describe slices and a slice is always readable. A destination that is
/// not memory is not.
#[derive(Clone, Copy)]
pub struct WriteOnly;

/// The visibility markers an [`ErasedTensor`] accepts.
pub trait ErasedIo: Clone + Copy + Send + Sync + 'static {}

/// A visibility that permits reads.
pub trait ErasedIoRead: ErasedIo {}

/// A visibility that permits writes.
pub trait ErasedIoWrite: ErasedIo {}

impl ErasedIo for ReadOnly {}
impl ErasedIo for ReadWrite {}
impl ErasedIo for WriteOnly {}

impl ErasedIoRead for ReadOnly {}
impl ErasedIoRead for ReadWrite {}

impl ErasedIoWrite for ReadWrite {}
impl ErasedIoWrite for WriteOnly {}

/// A tensor that has forgotten its line width. See the [module docs](crate::tensor).
pub struct ErasedTensor<E: Numeric, IO = ReadOnly> {
    _e: PhantomData<E>,
    _p: PhantomData<IO>,
}

/// Expand type for [`ErasedTensor`].
pub struct ErasedTensorExpand<E: Numeric, IO> {
    state: Arc<UnsafeCell<dyn ErasedTensorOperationsExpand<E>>>,
    _p: PhantomData<IO>,
}

impl<E: Numeric, IO> Clone for ErasedTensor<E, IO> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Numeric, IO> Copy for ErasedTensor<E, IO> {}

impl<E: Numeric, IO> Clone for ErasedTensorExpand<E, IO> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            _p: PhantomData,
        }
    }
}

/// What a backing has to answer to be an [`ErasedTensor`].
///
/// Expand-only, and that is the whole point: the value crosses as an
/// [`ExpandValue`], which is untyped in the IR, so the trait can be object-safe
/// without the width appearing in it. The implementor knows its own width and is
/// the one that can check the caller's against it.
///
/// Only the two width-free queries are required. A backing implements
/// [`__expand_read_line_method`](Self::__expand_read_line_method) or
/// [`__expand_write_line_method`](Self::__expand_write_line_method) for the half
/// it can serve, and declares it with [`ReadsLines`] / [`WritesLines`]; the
/// defaults are unreachable for a backing that declares itself honestly.
///
/// # Invariant
///
/// The line methods take and return an untyped [`ExpandValue`], so nothing in
/// their signature ties the value to a width. Every caller in this module checks
/// the width against [`__expand_vector_size_method`](Self::__expand_vector_size_method)
/// first, and a caller that skips the check builds IR whose type is a lie.
pub trait ErasedTensorOperationsExpand<E: Numeric> {
    /// The line width this backing takes, in scalars.
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize;

    /// How many lines the backing holds, which is what a *checked* access is
    /// checked against.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize>;

    /// Read one line at `index`, counted in lines of
    /// [`vector_size`](Self::__expand_vector_size_method).
    fn __expand_read_line_method(
        &self,
        _scope: &Scope,
        _index: NativeExpand<usize>,
    ) -> ExpandValue {
        unimplemented!("ErasedTensor: this backing does not serve reads")
    }

    /// Write one line at `index`, counted in lines of
    /// [`vector_size`](Self::__expand_vector_size_method).
    fn __expand_write_line_method(
        &mut self,
        _scope: &Scope,
        _index: NativeExpand<usize>,
        _value: ExpandValue,
    ) {
        unimplemented!("ErasedTensor: this backing does not serve writes")
    }
}

/// A backing whose [`__expand_read_line_method`](ErasedTensorOperationsExpand::__expand_read_line_method)
/// is real rather than the default.
pub trait ReadsLines<E: Numeric>: ErasedTensorOperationsExpand<E> {}

/// A backing whose [`__expand_write_line_method`](ErasedTensorOperationsExpand::__expand_write_line_method)
/// is real rather than the default.
pub trait WritesLines<E: Numeric>: ErasedTensorOperationsExpand<E> {}

/// A backing that can be wrapped at visibility `IO`.
///
/// This is what makes the `IO` marker mean something: it is implemented only
/// where the backing actually has the halves the marker promises, so the
/// constructors reject a mismatch instead of deferring it to a panic at the
/// first access.
pub trait ErasedBacking<E: Numeric, IO>: ErasedTensorOperationsExpand<E> {}

impl<E: Numeric, T: ReadsLines<E>> ErasedBacking<E, ReadOnly> for T {}
impl<E: Numeric, T: WritesLines<E>> ErasedBacking<E, WriteOnly> for T {}
impl<E: Numeric, T: ReadsLines<E> + WritesLines<E>> ErasedBacking<E, ReadWrite> for T {}

impl<E: Numeric, IO> ErasedTensorExpand<E, IO> {
    /// Erase `backing`'s width.
    ///
    /// Takes the *expand*, because that is where a width is known: the caller
    /// has the `N`-typed backing in hand and gives up naming it here.
    pub fn new<S: ErasedBacking<E, IO> + 'static>(backing: S) -> Self {
        Self {
            state: Arc::new(UnsafeCell::new(backing)),
            _p: PhantomData,
        }
    }

    fn state_read(&self) -> &dyn ErasedTensorOperationsExpand<E> {
        // SAFETY: the state is valid for the whole lifetime of `self`, and this
        // hands out a shared reference only.
        unsafe { &*self.state.get() }
    }

    #[allow(clippy::mut_from_ref)]
    fn state_write(&self) -> &mut dyn ErasedTensorOperationsExpand<E> {
        // SAFETY: as `VirtualTensorExpand`: the state is a handle into memory
        // and only the memory is written; the state itself is never mutated.
        unsafe { &mut *self.state.get() }
    }

    /// Panics unless the width `N` the caller is working at is the one the
    /// backing was built for.
    ///
    /// The alternative is an access that lands on a different element than the
    /// caller named, which no runtime check would catch, so this is a hard
    /// failure at expansion rather than a diagnostic.
    fn check_width<N: Size>(&self, scope: &Scope, op: &str) {
        let served = self.state_read().__expand_vector_size_method(scope);
        let asked = <N as Size>::__expand_value(scope);
        assert_eq!(
            served, asked,
            "ErasedTensor::{op}: the tensor takes {served}-wide lines and the {op} is {asked}-wide"
        );
    }
}

#[cube]
impl<E: Numeric, IO: ErasedIoRead> ErasedTensor<E, IO> {
    /// Read one `N`-wide line at `index`, counted in lines.
    ///
    /// # Panics
    ///
    /// At expansion, when `N` is not the width the tensor was built for.
    #[allow(unused)]
    pub fn read<N: Size>(&self, index: usize) -> Vector<E, N> {
        intrinsic!(|scope| {
            self.check_width::<N>(scope, "read");
            self.state_read()
                .__expand_read_line_method(scope, index)
                .into()
        })
    }
}

#[cube]
impl<E: Numeric, IO: ErasedIoWrite> ErasedTensor<E, IO> {
    /// Write one `N`-wide line at `index`, counted in lines.
    ///
    /// # Panics
    ///
    /// At expansion, when `N` is not the width the tensor was built for.
    #[allow(unused)]
    pub fn write<N: Size>(&mut self, index: usize, value: Vector<E, N>) {
        intrinsic!(|scope| {
            self.check_width::<N>(scope, "write");
            self.state_write()
                .__expand_write_line_method(scope, index, value.into())
        })
    }
}

#[cube]
impl<E: Numeric, IO: ErasedIo> ErasedTensor<E, IO> {
    /// How many lines the tensor holds.
    #[allow(unused, clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        intrinsic!(|scope| self.state_read().__expand_lines_method(scope))
    }
}

impl<E: Numeric, IO: ErasedIo> Vectorized for ErasedTensor<E, IO> {}
impl<E: Numeric, IO: ErasedIo> VectorizedExpand for ErasedTensorExpand<E, IO> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.state_read().__expand_vector_size_method(scope)
    }
}

// -- Backings ---------------------------------------------------------------

/// A [`VirtualTensor`] is the backing an erased tensor most often wraps, so it
/// is one without the caller writing an adapter.
///
/// The write half is declared only for [`ReadWrite`], which is the same gate
/// [`VirtualTensor`] itself puts on `write`.
impl<E: Numeric, N: Size, IO: Clone> ErasedTensorOperationsExpand<E>
    for VirtualTensorExpand<E, N, IO>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        VectorizedExpand::__expand_vector_size_method(self, scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.clone().__expand_len_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        Self::__expand_read_method(self, scope, index).expand
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
}

impl<E: Numeric, N: Size, IO: Clone> ReadsLines<E> for VirtualTensorExpand<E, N, IO> {}
impl<E: Numeric, N: Size> WritesLines<E> for VirtualTensorExpand<E, N, ReadWrite> {}

/// A launched tensor, reached through the indirection anyway.
///
/// The degenerate case, and the one a test compares the interesting cases
/// against.
impl<E: Numeric, N: Size> ErasedTensorOperationsExpand<E> for TensorExpand<Vector<E, N>> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        VectorizedExpand::__expand_vector_size_method(self, scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        unsafe {
            self.__expand_get_unchecked_method(scope, index)
                .__expand_deref_method(scope)
                .expand
        }
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
}

impl<E: Numeric, N: Size> ReadsLines<E> for TensorExpand<Vector<E, N>> {}
impl<E: Numeric, N: Size> WritesLines<E> for TensorExpand<Vector<E, N>> {}

/// Any view, at a width named once here and forgotten after.
///
/// The general way in, and the reason the other two are conveniences rather than
/// the interface: a backing that can already take a line at [`Coords1d`] is an
/// erased tensor, whatever it does with it. What this adds is only the erasure —
/// `N` is captured at construction and reappears when the access is made.
pub struct ErasedView<V, N: Size> {
    view: V,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size, V> ErasedTensorOperationsExpand<E> for ErasedView<V, N>
where
    V: ViewOperationsExpand<Vector<E, N>, Coords1d>,
{
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(_scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.view.__expand_shape_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        self.view.__expand_read_method(scope, index).expand
    }
}

impl<E: Numeric, N: Size, V> ReadsLines<E> for ErasedView<V, N> where
    V: ViewOperationsExpand<Vector<E, N>, Coords1d>
{
}

/// Any mutable view. As [`ErasedView`], with the write half as well.
pub struct ErasedViewMut<V, N: Size> {
    view: V,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size, V> ErasedTensorOperationsExpand<E> for ErasedViewMut<V, N>
where
    V: ViewOperationsMutExpand<Vector<E, N>, Coords1d>,
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        <V as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_shape_method(
            &self.view, scope,
        )
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        <V as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_read_method(
            &self.view, scope, index,
        )
        .expand
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        self.view.__expand_write_method(scope, index, value.into())
    }
}

impl<E: Numeric, N: Size, V> ReadsLines<E> for ErasedViewMut<V, N> where
    V: ViewOperationsMutExpand<Vector<E, N>, Coords1d>
{
}
impl<E: Numeric, N: Size, V> WritesLines<E> for ErasedViewMut<V, N> where
    V: ViewOperationsMutExpand<Vector<E, N>, Coords1d>
{
}

// -- Constructors -----------------------------------------------------------

impl<E: Numeric, IO: ErasedIo> ErasedTensor<E, IO> {
    /// The erased tensor over `view`, whose lines are `N` wide.
    ///
    /// The width is named at this call and nowhere after it, which is the whole
    /// point of the type: a caller that knows `N` hands it over here so the
    /// engine holding the tensor never has to.
    pub fn of_view<V: CubeType, N: Size>(_view: V) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_view`](Self::of_view).
    pub fn __expand_of_view<V: CubeType, N: Size>(
        _scope: &Scope,
        view: V::ExpandType,
    ) -> ErasedTensorExpand<E, IO>
    where
        V::ExpandType: ViewOperationsExpand<Vector<E, N>, Coords1d> + 'static,
        ErasedView<V::ExpandType, N>: ErasedBacking<E, IO>,
    {
        ErasedTensorExpand::new(ErasedView::<V::ExpandType, N> {
            view,
            _n: PhantomData,
        })
    }

    /// The erased tensor over the mutable `view`, whose lines are `N` wide.
    pub fn of_view_mut<V: CubeType, N: Size>(_view: V) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_view_mut`](Self::of_view_mut).
    pub fn __expand_of_view_mut<V: CubeType, N: Size>(
        _scope: &Scope,
        view: V::ExpandType,
    ) -> ErasedTensorExpand<E, IO>
    where
        V::ExpandType: ViewOperationsMutExpand<Vector<E, N>, Coords1d> + 'static,
        ErasedViewMut<V::ExpandType, N>: ErasedBacking<E, IO>,
    {
        ErasedTensorExpand::new(ErasedViewMut::<V::ExpandType, N> {
            view,
            _n: PhantomData,
        })
    }

    /// The erased tensor over `tensor` — memory, through the indirection.
    pub fn of_tensor<N: Size>(_tensor: &Tensor<Vector<E, N>>) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_tensor`](Self::of_tensor).
    pub fn __expand_of_tensor<N: Size>(
        _scope: &Scope,
        tensor: &TensorExpand<Vector<E, N>>,
    ) -> ErasedTensorExpand<E, IO>
    where
        TensorExpand<Vector<E, N>>: ErasedBacking<E, IO>,
    {
        ErasedTensorExpand::new(ExpandTypeClone::clone_unchecked(tensor))
    }

    /// The erased tensor over the mutable `tensor`.
    ///
    /// Separate from [`of_tensor`](Self::of_tensor) for the same reason
    /// [`of_view_mut`](Self::of_view_mut) is separate from
    /// [`of_view`](Self::of_view): a `&mut` operand expands to a `&mut` expand,
    /// and the shared entry point cannot take one.
    pub fn of_tensor_mut<N: Size>(_tensor: &mut Tensor<Vector<E, N>>) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_tensor_mut`](Self::of_tensor_mut).
    pub fn __expand_of_tensor_mut<N: Size>(
        _scope: &Scope,
        tensor: &mut TensorExpand<Vector<E, N>>,
    ) -> ErasedTensorExpand<E, IO>
    where
        TensorExpand<Vector<E, N>>: ErasedBacking<E, IO>,
    {
        ErasedTensorExpand::new(ExpandTypeClone::clone_unchecked(tensor))
    }

    /// The erased tensor over `tensor`.
    pub fn of_virtual<N: Size, IO2: Clone>(_tensor: VirtualTensor<E, N, IO2>) -> Self {
        unexpanded!()
    }

    /// Expand function for [`of_virtual`](Self::of_virtual).
    pub fn __expand_of_virtual<N: Size, IO2: Clone + 'static>(
        _scope: &Scope,
        tensor: VirtualTensorExpand<E, N, IO2>,
    ) -> ErasedTensorExpand<E, IO>
    where
        VirtualTensorExpand<E, N, IO2>: ErasedBacking<E, IO>,
    {
        ErasedTensorExpand::new(tensor)
    }
}

impl<E: Numeric, N: Size> From<VirtualTensorExpand<E, N, ReadWrite>>
    for ErasedTensorExpand<E, ReadWrite>
{
    fn from(tensor: VirtualTensorExpand<E, N, ReadWrite>) -> Self {
        ErasedTensorExpand::new(tensor)
    }
}

impl<E: Numeric, N: Size> From<VirtualTensorExpand<E, N, ReadOnly>>
    for ErasedTensorExpand<E, ReadOnly>
{
    fn from(tensor: VirtualTensorExpand<E, N, ReadOnly>) -> Self {
        ErasedTensorExpand::new(tensor)
    }
}

/// Making [`ErasedTensor`] a proper [cube type](CubeType), the same way
/// [`VirtualTensor`] is one.
mod __cube_type {
    use super::*;

    impl<E: Numeric, IO: ErasedIo> CubeType for ErasedTensor<E, IO> {
        type ExpandType = ErasedTensorExpand<E, IO>;
    }

    impl<E: Numeric, IO> IntoExpand for ErasedTensorExpand<E, IO> {
        type Expand = ErasedTensorExpand<E, IO>;

        fn into_expand(self, _: &Scope) -> Self::Expand {
            self
        }
    }

    impl<E: Numeric, IO> ExpandTypeClone for ErasedTensorExpand<E, IO> {
        fn clone_unchecked(&self) -> Self {
            self.clone()
        }
    }

    impl<E: Numeric, IO> IntoMut for ErasedTensorExpand<E, IO> {
        fn into_mut(self, _scope: &Scope) -> Self {
            self
        }
    }

    impl<E: Numeric, IO> CubeDebug for ErasedTensorExpand<E, IO> {}

    impl<E: Numeric, IO> AsRefExpand for ErasedTensorExpand<E, IO> {
        fn __expand_ref_method(&self, _: &Scope) -> &Self {
            self
        }
    }

    impl<E: Numeric, IO> AsMutExpand for ErasedTensorExpand<E, IO> {
        fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
            self
        }
    }
}

// -- As the backing of a view -----------------------------------------------

/// The erased tensor as the backing of a [`View`](crate::tensor::View) or a
/// [`ViewMut`](crate::tensor::ViewMut), which is how a kernel that addresses its operand through a
/// layout reaches one.
///
/// `N` is constrained by the trait rather than by the type, which is the whole
/// trick: the tensor stays width-free, and the width arrives with the view the
/// caller builds over it. Every access checks the two agree.
///
/// This is implemented for every visibility, including [`WriteOnly`], because
/// [`ViewOperationsMut`] has [`ViewOperations`] as a supertrait — a write-only
/// destination backing a [`ViewMut`](crate::tensor::ViewMut) still has to name the read
/// half. Its reads
/// reach the backing's default and panic.
impl<E: Numeric, N: Size, IO: ErasedIo> ViewOperations<Vector<E, N>, Coords1d>
    for ErasedTensor<E, IO>
{
}

impl<E: Numeric, N: Size, IO: ErasedIo> ViewOperationsExpand<Vector<E, N>, Coords1d>
    for ErasedTensorExpand<E, IO>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        self.check_width::<N>(scope, "read");
        self.state_read()
            .__expand_read_line_method(scope, pos)
            .into()
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        let zero = <Vector<E, N>>::__expand_cast_from(scope, 0.into());
        <Self as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_read_masked_method(
            self, scope, pos, zero,
        )
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        mask_value: <Vector<E, N> as CubeType>::ExpandType,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        let in_bounds =
            <Self as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_is_in_bounds_method(
                self, scope, pos,
            );
        // Fold an out-of-bounds index to 0 before reading, as
        // `cubecl_core::io::read_masked` does, so the read itself is in bounds.
        let keep = usize::__expand_cast_from(scope, in_bounds);
        let pos = pos.__expand_mul_method(scope, keep);
        let value = <Self as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_read_method(
            self, scope, pos,
        );
        select::expand::<Vector<E, N>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <Vector<E, N> as CubeType>::ExpandType {
        <Self as ViewOperationsExpand<Vector<E, N>, Coords1d>>::__expand_read_method(
            self, scope, pos,
        )
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: NativeExpand<usize>,
        _end: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<E, N>> {
        unimplemented!("ErasedTensor: no slice yet, see the module docs")
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
        unimplemented!("ErasedTensor: not a tensor map")
    }
}

impl<E: Numeric, N: Size, IO: ErasedIoWrite> ViewOperationsMut<Vector<E, N>, Coords1d>
    for ErasedTensor<E, IO>
{
}

impl<E: Numeric, N: Size, IO: ErasedIoWrite> ViewOperationsMutExpand<Vector<E, N>, Coords1d>
    for ErasedTensorExpand<E, IO>
{
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <Vector<E, N> as CubeType>::ExpandType,
    ) {
        self.check_width::<N>(scope, "write");
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
        unimplemented!("ErasedTensor: no slice yet, see the module docs")
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &Scope,
        _shared_memory: &SliceExpand<Vector<E, N>>,
        _pos: NativeExpand<usize>,
    ) {
        unimplemented!("ErasedTensor: not a tensor map")
    }
}
