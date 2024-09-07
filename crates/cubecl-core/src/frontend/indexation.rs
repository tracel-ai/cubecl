use super::{CubeType, ExpandElement, ExpandElementTyped};
use crate::{
    ir::{IntKind, Variable},
    unexpanded,
};

/// Fake indexation so we can rewrite indexes into scalars as calls to this fake function in the
/// non-expanded function
pub trait CubeIndex<T: Index> {
    type Output: CubeType;

    fn cube_idx(&self, _i: T) -> &Self::Output {
        unexpanded!()
    }
}

pub trait CubeIndexMut<T: Index>: CubeIndex<T> {
    fn cube_idx_mut(&mut self, _i: T) -> &mut Self::Output {
        unexpanded!()
    }
}

pub trait Index {
    fn value(self) -> Variable;
}

impl Index for i32 {
    fn value(self) -> Variable {
        Variable::ConstantScalar(crate::ir::ConstantScalarValue::Int(
            self as i64,
            IntKind::I32,
        ))
    }
}

impl Index for u32 {
    fn value(self) -> Variable {
        Variable::ConstantScalar(crate::ir::ConstantScalarValue::UInt(self as u64))
    }
}

impl Index for ExpandElement {
    fn value(self) -> Variable {
        *self
    }
}

impl Index for ExpandElementTyped<u32> {
    fn value(self) -> Variable {
        *self.expand
    }
}
