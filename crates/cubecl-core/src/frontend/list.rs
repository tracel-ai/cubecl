use core::ops::{Deref, DerefMut};

use super::{CubeType, NativeExpand};
use crate as cubecl;
use crate::{prelude::*, unexpanded};
use cubecl_ir::{Scope, VectorSize};

/// Type from which we can read values in cube functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[cube(self_type = "ref", expand_base_traits = "SliceOperatorExpand<T>")]
pub trait List<T: CubePrimitive>: SliceOperator<T> + Vectorized + Deref<Target = [T]> {
    #[allow(unused)]
    fn read(&self, index: usize) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, index: usize) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn len(&self) -> usize {
        unexpanded!();
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[cube(self_type = "ref", expand_base_traits = "SliceMutOperatorExpand<T>")]
pub trait ListMut<T: CubePrimitive>:
    List<T> + SliceMutOperator<T> + DerefMut<Target = [T]>
{
    #[allow(unused)]
    fn write(&self, index: usize, value: T) {
        unexpanded!()
    }
}

// Automatic implementation for references to List.
impl<'a, T: CubePrimitive, L: List<T>> List<T> for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
    &'a L: Deref<Target = [T]>,
{
    fn read(&self, index: usize) -> T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for mutable references to List.
impl<'a, T: CubePrimitive, L: List<T>> List<T> for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
    &'a mut L: Deref<Target = [T]>,
{
    fn read(&self, index: usize) -> T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for references to ListMut.
impl<'a, T: CubePrimitive, L: ListMut<T>> ListMut<T> for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
    &'a L: DerefMut<Target = [T]>,
{
    fn write(&self, index: usize, value: T) {
        L::write(self, index, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, index, value);
    }
}

// Automatic implementation for mutable references to ListMut.
impl<'a, T: CubePrimitive, L: ListMut<T>> ListMut<T> for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
    &'a mut L: DerefMut<Target = [T]>,
{
    fn write(&self, index: usize, value: T) {
        L::write(self, index, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: NativeExpand<usize>,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, index, value);
    }
}

pub trait Vectorized: CubeType<ExpandType: VectorizedExpand> {
    fn vector_size(&self) -> VectorSize {
        unexpanded!()
    }
    fn __expand_vector_size(_scope: &mut Scope, this: Self::ExpandType) -> VectorSize {
        this.vector_size()
    }
}

pub trait VectorizedExpand {
    fn vector_size(&self) -> VectorSize;
    fn __expand_vector_size_method(&self, _scope: &mut Scope) -> VectorSize {
        self.vector_size()
    }
}

impl<'a, L: Vectorized> Vectorized for &'a L where &'a L: CubeType<ExpandType: VectorizedExpand> {}
impl<'a, L: Vectorized> Vectorized for &'a mut L where
    &'a mut L: CubeType<ExpandType: VectorizedExpand>
{
}
