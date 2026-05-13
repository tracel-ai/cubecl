use crate as cubecl;
use crate::prelude::CubePrimitive;
use cubecl::frontend::assign::expand_no_check;
use cubecl::prelude::*;
use cubecl_ir::Operation;
use cubecl_macros::intrinsic;

#[derive(Clone, Copy)]
pub struct RuntimeCell<T: CubeType> {
    #[allow(unused)]
    value: T,
}

pub struct RuntimeCellExpand<T: CubeType> {
    value: <T as cubecl::prelude::CubeType>::ExpandType,
}
impl<T: CubeType<ExpandType: Clone>> Clone for RuntimeCellExpand<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

impl<T: CubeType> IntoExpand for RuntimeCellExpand<T> {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self
    }
}
impl<T: CubeType> ExpandTypeClone for RuntimeCellExpand<T> {
    fn clone_unchecked(&self) -> Self {
        Self {
            value: self.value.clone_unchecked(),
        }
    }
}
impl<T: CubeType> cubecl::prelude::CubeType for RuntimeCell<T> {
    type ExpandType = RuntimeCellExpand<T>;
}

impl<T: CubeType> cubecl::prelude::IntoMut for RuntimeCellExpand<T> {
    fn into_mut(self, _scope: &Scope) -> Self {
        Self {
            // We keep the same as a cell would do.
            value: self.value,
        }
    }
}
impl<T: CubeType> cubecl::prelude::CubeDebug for RuntimeCellExpand<T> {}

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
    #[allow(unused_variables)]
    pub fn store(&self, value: T) {
        intrinsic!(|scope| {
            let mut this = self.value.clone();
            expand_no_check(scope, value, &mut this);
        })
    }

    /// Get the value from the call
    pub fn read(&self) -> T {
        intrinsic!(|scope| {
            let value = init_expand(scope, self.value.clone().expand, false, Operation::Copy);
            value.into()
        })
    }

    /// Consume the cell.
    pub fn consume(self) -> T {
        intrinsic!(|scope| { self.value })
    }
}

#[cube]
impl<S: Scalar, N: Size> RuntimeCell<Vector<S, N>> {
    /// Extract the value in the cell at the given index.
    #[allow(unused_variables)]
    pub fn extract(&mut self, index: usize) -> S {
        intrinsic!(|scope| { self.value.__expand_extract_method(scope, index) })
    }

    /// Store a new value in the cell at the given index.
    #[allow(unused_variables)]
    pub fn insert(&mut self, index: usize, value: S) {
        intrinsic!(|scope| { self.value.__expand_insert_method(scope, index, value) })
    }
}

impl<T: CubeType> AsRefExpand for RuntimeCellExpand<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<T: CubeType> AsMutExpand for RuntimeCellExpand<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
