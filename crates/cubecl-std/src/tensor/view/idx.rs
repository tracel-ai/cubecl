use cubecl_core::prelude::*;

use crate::tensor::{View, ViewExpand, layout::Coordinates};

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeIndex for View<E, C, IO> {
    type Output = E;
    type Idx = C;
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeIndexExpand for ViewExpand<E, C, IO> {
    type Output = <E as CubeType>::ExpandType;
    type Idx = <C as CubeType>::ExpandType;

    fn expand_index(self, scope: &mut Scope, index: C::ExpandType) -> Self::Output {
        self.__expand_read_method(scope, index)
    }

    fn expand_index_unchecked(self, scope: &mut Scope, index: C::ExpandType) -> Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: CubePrimitive, C: Coordinates> CubeIndexMut for View<E, C, ReadWrite> {}
impl<E: CubePrimitive, C: Coordinates> CubeIndexMutExpand for ViewExpand<E, C, ReadWrite> {
    fn expand_index_mut(self, scope: &mut Scope, index: C::ExpandType, value: Self::Output) {
        self.__expand_write_method(scope, index, value)
    }
}
