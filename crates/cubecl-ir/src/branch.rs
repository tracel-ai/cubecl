use alloc::{boxed::Box, format, vec::Vec};
use core::fmt::Display;

use crate::OperationReflect;

use super::{ElemType, OperationCode, Scope, Type, UIntKind, Variable};
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

impl If {
    /// Registers an if statement to the given scope.
    pub fn register<F: Fn(&mut Scope)>(parent_scope: &mut Scope, cond: Variable, func: F) {
        let mut scope = parent_scope.child();

        func(&mut scope);

        let op = Self { cond, scope };
        parent_scope.register(Branch::If(Box::new(op)));
    }
}

impl IfElse {
    /// Registers an if else statement to the given scope.
    pub fn register<IF, ELSE>(
        parent_scope: &mut Scope,
        cond: Variable,
        func_if: IF,
        func_else: ELSE,
    ) where
        IF: Fn(&mut Scope),
        ELSE: Fn(&mut Scope),
    {
        let mut scope_if = parent_scope.child();
        let mut scope_else = parent_scope.child();

        func_if(&mut scope_if);
        func_else(&mut scope_else);

        parent_scope.register(Branch::IfElse(Box::new(Self {
            cond,
            scope_if,
            scope_else,
        })));
    }
}

impl RangeLoop {
    /// Registers a range loop to the given scope.
    pub fn register<F: Fn(Variable, &mut Scope)>(
        parent_scope: &mut Scope,
        start: Variable,
        end: Variable,
        step: Option<Variable>,
        inclusive: bool,
        func: F,
    ) {
        let mut scope = parent_scope.child();
        let index_ty = Type::scalar(ElemType::UInt(UIntKind::U32));
        let i = *scope.create_local_restricted(index_ty);

        func(i, &mut scope);

        parent_scope.register(Branch::RangeLoop(Box::new(Self {
            i,
            start,
            end,
            step,
            scope,
            inclusive,
        })));
    }
}

impl Loop {
    /// Registers a loop to the given scope.
    pub fn register<F: Fn(&mut Scope)>(parent_scope: &mut Scope, func: F) {
        let mut scope = parent_scope.child();

        func(&mut scope);

        let op = Self { scope };
        parent_scope.register(Branch::Loop(Box::new(op)));
    }
}

#[allow(missing_docs)]
pub struct UnrolledRangeLoop;

impl UnrolledRangeLoop {
    /// Registers an unrolled range loop to the given scope.
    pub fn register<F: Fn(Variable, &mut Scope)>(
        scope: &mut Scope,
        start: u32,
        end: u32,
        step: Option<u32>,
        inclusive: bool,
        func: F,
    ) {
        if inclusive {
            if let Some(step) = step {
                for i in (start..=end).step_by(step as usize) {
                    func(i.into(), scope);
                }
            } else {
                for i in start..=end {
                    func(i.into(), scope);
                }
            }
        } else if let Some(step) = step {
            for i in (start..end).step_by(step as usize) {
                func(i.into(), scope);
            }
        } else {
            for i in start..end {
                func(i.into(), scope);
            }
        }
    }
}
