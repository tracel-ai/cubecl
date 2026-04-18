use core::ops::{Deref, DerefMut};

use super::CubeType;
use crate as cubecl;
use crate::{prelude::*, unexpanded};
use cubecl_ir::{Scope, VectorSize};

/// Type from which we can read values in cube functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[cube(expand_base_traits = "SliceOperatorExpand<'a, T>")]
pub trait List<'a, T: CubePrimitive>:
    SliceOperator<'a, T> + Vectorized + Deref<Target = [T]>
{
    #[allow(unused)]
    fn read(&'a self, index: usize) -> &'a T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&'a self, index: usize) -> &'a T {
        unexpanded!()
    }

    #[allow(unused)]
    fn len(&self) -> usize {
        unexpanded!();
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[cube(expand_base_traits = "SliceMutOperatorExpand<'a, T>")]
pub trait ListMut<'a, T: CubePrimitive>:
    List<'a, T> + SliceMutOperator<'a, T> + DerefMut<Target = [T]>
{
    #[allow(unused)]
    fn write(&'a self, index: usize) -> &'a mut T {
        unexpanded!()
    }
}

// Automatic implementation for references to List.
impl<'a, T: CubePrimitive, L: List<'a, T>> List<'a, T> for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
    &'a L: Deref<Target = [T]>,
{
    fn read(&self, index: usize) -> &T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: &'a Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> &'a <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for mutable references to List.
impl<'a, T: CubePrimitive, L: List<'a, T>> List<'a, T> for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
    &'a mut L: Deref<Target = [T]>,
{
    fn read(&'a self, index: usize) -> &'a T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: &'a Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> &'a <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for references to ListMut.
impl<'a, T: CubePrimitive, L: ListMut<'a, T>> ListMut<'a, T> for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
    &'a L: DerefMut<Target = [T]>,
{
    fn write(&'a self, index: usize) -> &'a mut T {
        L::write(self, index)
    }

    fn __expand_write(
        scope: &mut Scope,
        this: &'a Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> &'a mut T::ExpandType {
        L::__expand_write(scope, this, index)
    }
}

// Automatic implementation for mutable references to ListMut.
impl<'a, T: CubePrimitive, L: ListMut<'a, T>> ListMut<'a, T> for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
    &'a mut L: DerefMut<Target = [T]>,
{
    fn write(&'a self, index: usize) -> &'a mut T {
        L::write(self, index)
    }

    fn __expand_write(
        scope: &mut Scope,
        this: &'a Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> &'a mut T::ExpandType {
        L::__expand_write(scope, this, index)
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
