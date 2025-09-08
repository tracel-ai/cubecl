use super::{CubeType, ExpandElementTyped};
use crate::{
    prelude::{
        CubePrimitive, SliceMutOperator, SliceMutOperatorExpand, SliceOperator, SliceOperatorExpand,
    },
    unexpanded,
};
use cubecl_ir::Scope;

/// Type from which we can read values in cube functions.
/// For a mutable version, see [ListMut].
#[allow(clippy::len_without_is_empty)]
pub trait List<T: CubePrimitive>:
    CubeType<ExpandType: ListExpand<T> + SliceOperatorExpand<T>> + SliceOperator<T>
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

    fn line_size(&self) -> u32 {
        unexpanded!();
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        this.__expand_read_method(scope, index)
    }

    fn __expand_read_unchecked(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        this.__expand_read_unchecked_method(scope, index)
    }

    fn __expand_len(scope: &mut Scope, this: Self::ExpandType) -> ExpandElementTyped<u32> {
        this.__expand_len_method(scope)
    }

    fn __expand_line_size(scope: &mut Scope, this: Self::ExpandType) -> u32 {
        this.__expand_line_size_method(scope)
    }
}

/// Expand version of [CubeRead].
pub trait ListExpand<T: CubePrimitive>: SliceOperatorExpand<T> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType;
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType;
    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32>;
    fn line_size(&self) -> u32;
    fn __expand_line_size_method(&self, scope: &mut Scope) -> u32;
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
pub trait ListMut<T: CubePrimitive>:
    CubeType<ExpandType: ListMutExpand<T> + SliceMutOperatorExpand<T>> + List<T> + SliceMutOperator<T>
{
    #[allow(unused)]
    fn write(&self, index: u32, value: T) {
        unexpanded!()
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    ) {
        this.__expand_write_method(scope, index, value)
    }
}

/// Expand version of [CubeWrite].
pub trait ListMutExpand<T: CubePrimitive>: ListExpand<T> + SliceMutOperatorExpand<T> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    );
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
