use alloc::{boxed::Box, format, vec::Vec};
use core::fmt::Display;

use crate::OperationReflect;

use super::{OperationCode, Scope, Variable};
use crate::TypeHash;

/// All branching types.
#[allow(clippy::large_enum_variant)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = BranchOpCode)]
pub enum Branch {
    /// An if statement.
    If(Box<If>),
    /// An if else statement.
    IfElse(Box<IfElse>),
    /// A switch statement
    Switch(Box<Switch>),
    /// A range loop.
    RangeLoop(Box<RangeLoop>),
    /// A loop.
    Loop(Box<Loop>),
    /// A return statement.
    Return,
    /// A break statement.
    Break,
}

impl OperationReflect for Branch {
    type OpCode = BranchOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }
}

impl Display for Branch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Branch::If(if_) => write!(f, "if({}) {}", if_.cond, if_.scope),
            Branch::IfElse(if_else) => write!(
                f,
                "if({}) {} else {}",
                if_else.cond, if_else.scope_if, if_else.scope_else
            ),
            Branch::Switch(switch) => write!(
                f,
                "switch({}) {:?}",
                switch.value,
                switch
                    .cases
                    .iter()
                    .map(|case| format!("{}", case.0))
                    .collect::<Vec<_>>(),
            ),
            Branch::RangeLoop(range_loop) => write!(
                f,
                "for({} in {}{}{}) {}",
                range_loop.i,
                range_loop.start,
                if range_loop.inclusive { "..=" } else { ".." },
                range_loop.end,
                range_loop.scope
            ),
            Branch::Loop(loop_) => write!(f, "loop {}", loop_.scope),
            Branch::Return => write!(f, "return"),
            Branch::Break => write!(f, "break"),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct If {
    pub cond: Variable,
    pub scope: Scope,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct IfElse {
    pub cond: Variable,
    pub scope_if: Scope,
    pub scope_else: Scope,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct Switch {
    pub value: Variable,
    pub scope_default: Scope,
    pub cases: Vec<(Variable, Scope)>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct RangeLoop {
    pub i: Variable,
    pub start: Variable,
    pub end: Variable,
    pub step: Option<Variable>,
    pub inclusive: bool,
    pub scope: Scope,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct Loop {
    pub scope: Scope,
}
