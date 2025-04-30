use std::{cell::RefCell, rc::Rc};

use cubecl_ir::Scope;

use crate::prelude::{CubeDebug, CubeType, IntoMut};

#[derive(Debug, Clone)]
/// A cell that can store and mutate a cube type during comptime.
pub struct ComptimeCell<T: CubeType> {
    pub(super) value: Rc<RefCell<T>>,
}

/// Expand type of [ComptimeCell].
pub struct ComptimeCellExpand<T: CubeType> {
    // We clone the expand type during the compilation phase, but for register reuse, not for
    // copying data. To achieve the intended behavior, we have to share the same underlying values.
    pub(super) value: Rc<RefCell<T::ExpandType>>,
}

impl<T: CubeType> CubeType for ComptimeCell<T> {
    type ExpandType = ComptimeCellExpand<T>;
}

impl<T: CubeType + Clone> ComptimeCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Rc::new(RefCell::new(value)),
        }
    }
    pub fn __expand_new(_scope: &mut Scope, value: T::ExpandType) -> ComptimeCellExpand<T> {
        ComptimeCellExpand {
            value: Rc::new(RefCell::new(value)),
        }
    }
    pub fn read(&self) -> T {
        let value = self.value.borrow();
        value.clone()
    }
    pub fn store(&mut self, value: T) {
        let mut old = self.value.borrow_mut();
        *old = value;
    }
    pub fn __expand_store(context: &mut Scope, this: ComptimeCellExpand<T>, value: T::ExpandType) {
        this.__expand_store_method(context, value)
    }
    pub fn __expand_read(scope: &mut Scope, this: ComptimeCellExpand<T>) -> T::ExpandType {
        this.__expand_read_method(scope)
    }
}

impl<T: CubeType + Clone> ComptimeCellExpand<T> {
    pub fn __expand_store_method(&self, _context: &mut Scope, value: T::ExpandType) {
        let mut old = self.value.borrow_mut();
        *old = value;
    }
    pub fn __expand_read_method(&self, _scope: &mut Scope) -> T::ExpandType {
        let value = self.value.borrow();
        value.clone()
    }
}

impl<T: CubeType> IntoMut for ComptimeCellExpand<T> {
    fn into_mut(self, scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubeType> CubeDebug for ComptimeCellExpand<T> {}

impl<T: CubeType> Clone for ComptimeCellExpand<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}
