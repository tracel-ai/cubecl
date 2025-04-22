use super::{CubeType, ExpandElementTyped};
use crate::unexpanded;
use cubecl_ir::Scope;

/// Type from which we can read values in cube functions.
/// For a mutable version, see [ListMut].
pub trait List<T: CubeType>: CubeType<ExpandType: ListExpand<T>> {
    #[allow(unused)]
    fn read(&self, index: u32) -> T {
        unexpanded!()
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        this.__expand_read_method(scope, index)
    }
}

/// Expand version of [CubeRead].
pub trait ListExpand<T: CubeType> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType;
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
pub trait ListMut<T: CubeType>: CubeType<ExpandType: ListMutExpand<T>> + List<T> {
    #[allow(unused)]
    fn write(&self, index: u32, value: T) {
        unexpanded!()
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    );
}

/// Expand version of [CubeWrite].
pub trait ListMutExpand<T: CubeType>: ListExpand<T> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    );
}

// Automatic implementation for mutable references to List.
impl<'a, T: CubeType, L: List<T>> List<T> for &'a L
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
impl<'a, T: CubeType, L: List<T>> List<T> for &'a mut L
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
impl<'a, T: CubeType, L: ListMut<T>> ListMut<T> for &'a L
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

// Automatic implementation for references to ListMut.
impl<'a, T: CubeType, L: ListMut<T>> ListMut<T> for &'a mut L
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
