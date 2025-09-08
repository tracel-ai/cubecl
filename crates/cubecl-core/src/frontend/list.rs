use super::{CubeType, ExpandElementTyped};
use crate as cubecl;
use crate::{prelude::*, unexpanded};
use cubecl_ir::Scope;

/// Type from which we can read values in cube functions.
/// For a mutable version, see [ListMut].
#[allow(clippy::len_without_is_empty)]
#[cube(receiver_type = "ref")]
pub trait List<T: CubePrimitive>:
    CubeType<ExpandType: ListExpand<T> + SliceOperatorExpand<T>> + SliceOperator<T> + Lined
{
    #[allow(unused)]
    fn read(&self, index: u32) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, index: u32) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn len(&self) -> u32 {
        unexpanded!();
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[cube(receiver_type = "ref")]
pub trait ListMut<T: CubePrimitive>:
    CubeType<ExpandType: ListMutExpand<T> + SliceMutOperatorExpand<T>> + List<T> + SliceMutOperator<T>
{
    #[allow(unused)]
    fn write(&self, index: u32, value: T) {
        unexpanded!()
    }
}

// Automatic implementation for references to List.
impl<'a, T: CubePrimitive, L: List<T>> List<T> for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
{
    fn read(&self, index: u32) -> T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for mutable references to List.
impl<'a, T: CubePrimitive, L: List<T>> List<T> for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
{
    fn read(&self, index: u32) -> T {
        L::read(self, index)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, index)
    }
}

// Automatic implementation for references to ListMut.
impl<'a, T: CubePrimitive, L: ListMut<T>> ListMut<T> for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
{
    fn write(&self, index: u32, value: T) {
        L::write(self, index, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, index, value);
    }
}

// Automatic implementation for mutable references to ListMut.
impl<'a, T: CubePrimitive, L: ListMut<T>> ListMut<T> for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
{
    fn write(&self, index: u32, value: T) {
        L::write(self, index, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, index, value);
    }
}

pub trait Lined: CubeType<ExpandType: LinedExpand> {
    fn line_size(&self) -> u32 {
        unexpanded!()
    }
    fn __expand_line_size(_scope: &mut Scope, this: Self::ExpandType) -> u32 {
        this.line_size()
    }
}

pub trait LinedExpand {
    fn line_size(&self) -> u32;
    fn __expand_line_size_method(&self, _scope: &mut Scope) -> u32 {
        self.line_size()
    }
}

impl<'a, L: Lined> Lined for &'a L where &'a L: CubeType<ExpandType: LinedExpand> {}
impl<'a, L: Lined> Lined for &'a mut L where &'a mut L: CubeType<ExpandType: LinedExpand> {}
