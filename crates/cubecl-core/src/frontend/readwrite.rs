use super::{CubeType, ExpandElementTyped};
use crate::unexpanded;
use cubecl_ir::Scope;

/// Type from which we can read values in cube functions.
pub trait CubeRead<T: CubeType>: CubeType<ExpandType: CubeReadExpand<T>> {
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
pub trait CubeReadExpand<T: CubeType> {
    fn __expand_read_method(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T>;
}

/// Type into which we can write values in cube functions.
pub trait CubeWrite<T: CubeType>: CubeType<ExpandType: CubeWriteExpand<T>> {
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
pub trait CubeWriteExpand<T: CubeType> {
    fn __expand_write_method(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    );
}
