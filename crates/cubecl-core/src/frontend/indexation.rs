use super::{Comptime, ExpandElement, ExpandElementTyped, UInt};
use crate::ir::{IntKind, Variable};

pub trait Index {
    fn value(self) -> Variable;
}

impl Index for Comptime<u32> {
    fn value(self) -> Variable {
        Variable::ConstantScalar(crate::ir::ConstantScalarValue::UInt(self.inner as u64))
    }
}

impl Index for Comptime<i32> {
    fn value(self) -> Variable {
        Variable::ConstantScalar(crate::ir::ConstantScalarValue::Int(
            self.inner as i64,
            IntKind::I32,
        ))
    }
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

impl Index for UInt {
    fn value(self) -> Variable {
        Variable::ConstantScalar(crate::ir::ConstantScalarValue::UInt(self.val as u64))
    }
}

impl Index for ExpandElement {
    fn value(self) -> Variable {
        *self
    }
}

impl Index for ExpandElementTyped<UInt> {
    fn value(self) -> Variable {
        let value: ExpandElement = self.into();
        value.value()
    }
}
