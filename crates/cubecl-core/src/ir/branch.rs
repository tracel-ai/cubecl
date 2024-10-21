use std::fmt::Display;

use super::{Elem, Item, Scope, Variable};
use serde::{Deserialize, Serialize};

/// All branching types.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Branch {
    /// An if statement.
    If(Box<If>),
    /// An if else statement.
    IfElse(Box<IfElse>),
    // A select statement/ternary
    Select(Select),
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

impl Display for Branch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Branch::If(if_) => write!(f, "if({})", if_.cond),
            Branch::IfElse(if_else) => write!(f, "if({})", if_else.cond),
            Branch::Select(select) => write!(
                f,
                "{} = select({}, {}, {})",
                select.out, select.cond, select.then, select.or_else
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
                "for({} in {}{}{})",
                range_loop.i,
                range_loop.start,
                if range_loop.inclusive { "..=" } else { ".." },
                range_loop.end
            ),
            Branch::Loop(_) => write!(f, "loop{{}}"),
            Branch::Return => write!(f, "return"),
            Branch::Break => write!(f, "break"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct If {
    pub cond: Variable,
    pub scope: Scope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct IfElse {
    pub cond: Variable,
    pub scope_if: Scope,
    pub scope_else: Scope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct Select {
    pub cond: Variable,
    pub then: Variable,
    pub or_else: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct Switch {
    pub value: Variable,
    pub scope_default: Scope,
    pub cases: Vec<(Variable, Scope)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct RangeLoop {
    pub i: Variable,
    pub start: Variable,
    pub end: Variable,
    pub step: Option<Variable>,
    pub inclusive: bool,
    pub scope: Scope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
        let index_ty = Item::new(Elem::UInt);
        let i = scope.create_local_undeclared(index_ty);

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
