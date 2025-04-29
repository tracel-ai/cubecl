use crate as cubecl;
use crate::prelude::CubePrimitive;
use cubecl::prelude::*;
use cubecl_ir::Operation;
use cubecl_macros::intrinsic;

#[derive(CubeType, Clone, Copy)]
pub struct RuntimeCell<T: CubeType> {
    value: T,
}

#[cube]
impl<T: CubePrimitive> RuntimeCell<T> {
    /// Create a new runtime cell with the given initial value.
    #[allow(unused_variables)]
    pub fn new(init: T) -> Self {
        intrinsic!(|scope| {
            let value = init_expand(scope, init.expand, true, Operation::Copy);
            RuntimeCellExpand {
                value: value.into(),
            }
        })
    }

    /// Store a new value in the cell.
    pub fn store(&mut self, value: T) {
        self.value = value;
    }

    /// Get the value from the call
    pub fn read(&self) -> T {
        intrinsic!(|scope| {
            let value = init_expand(scope, self.value.expand, false, Operation::Copy);
            value.into()
        })
    }

    /// Consume the cell.
    pub fn consume(self) -> T {
        intrinsic!(|scope| { self.value })
    }
}

#[cube]
impl<T: CubeIndexMut> RuntimeCell<T> {
    /// Store a new value in the cell at the given index.
    #[allow(unused_variables)]
    pub fn store_at(&mut self, index: u32, value: T::Output) {
        intrinsic!(|scope| { self.value.expand_index_mut(scope, index, value) })
    }
}

#[cube]
impl<T: CubeIndex> RuntimeCell<T> {
    /// Read a value in the cell at the given index.
    #[allow(unused_variables)]
    pub fn read_at(&self, index: u32) -> T::Output {
        intrinsic!(|scope| { self.value.expand_index(scope, index) })
    }
}
