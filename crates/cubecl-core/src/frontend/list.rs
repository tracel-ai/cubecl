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
    ) -> ExpandElementTyped<T>;
}

/// Expand version of [CubeRead].
pub trait ListExpand<T: CubeType> {
    fn __expand_read_method(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T>;
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
        value: ExpandElementTyped<T>,
    );
}

/// Expand version of [CubeWrite].
pub trait ListMutExpand<T: CubeType> {
    fn __expand_write_method(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    );
}
