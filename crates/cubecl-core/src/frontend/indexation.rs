use super::ExpandElement;
use crate::ir::{IntKind, Variable};

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
