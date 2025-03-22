use alloc::collections::VecDeque;

use alloc::vec;
use alloc::vec::Vec;

use crate::Variable;

/// An operation that can be reflected on
pub trait OperationReflect: Sized {
    /// Type of the op codes for this operation
    type OpCode;

    /// Get the opcode for this operation
    fn op_code(&self) -> Self::OpCode;
    /// Get the list of arguments for this operation. If not all arguments are [`Variable`], returns
    /// `None` instead.
    fn args(&self) -> Option<Vec<Variable>> {
        None
    }
    /// Create typed operation from an opcode and a list of arguments. Returns `None` if not all
    /// arguments are [`Variable`].
    #[allow(unused)]
    fn from_code_and_args(op_code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        None
    }
    /// Whether this operation is commutative (arguments can be freely reordered). Ignored for
    /// single argument operations.
    fn is_commutative(&self) -> bool {
        false
    }
    /// Whether this operation is pure (has no side effects). Things like uniform/plane operations
    /// are considered impure, because they affect other units.
    fn is_pure(&self) -> bool {
        false
    }
}

/// A type that represents an operation's arguments
pub trait OperationArgs: Sized {
    /// Construct this type from a list of arguments. If not all arguments are [`Variable`], returns
    /// `None`
    #[allow(unused)]
    fn from_args(args: &[Variable]) -> Option<Self> {
        None
    }

    /// Turns this type into a flat list of arguments. If not all arguments are [`Variable`],
    /// returns `None`
    fn as_args(&self) -> Option<Vec<Variable>> {
        None
    }
}

impl OperationArgs for Variable {
    fn from_args(args: &[Variable]) -> Option<Self> {
        Some(args[0])
    }

    fn as_args(&self) -> Option<Vec<Variable>> {
        Some(vec![*self])
    }
}

/// Types that can be destructured into and created from a list of [`Variable`]s.
pub trait FromArgList: Sized {
    /// Creates this type from a list of variables. This works like a parse stream, where consumed
    /// variables are popped from the front.
    fn from_arg_list(args: &mut VecDeque<Variable>) -> Self;
    /// Turns this type into a list of [`Variable`]s.
    fn as_arg_list(&self) -> impl IntoIterator<Item = Variable>;
}

impl FromArgList for Variable {
    fn from_arg_list(args: &mut VecDeque<Variable>) -> Self {
        args.pop_front().expect("Missing variable from arg list")
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Variable> {
        [*self]
    }
}

impl FromArgList for Vec<Variable> {
    fn from_arg_list(args: &mut VecDeque<Variable>) -> Self {
        core::mem::take(args).into_iter().collect()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Variable> {
        self.iter().cloned()
    }
}

impl FromArgList for bool {
    fn from_arg_list(args: &mut VecDeque<Variable>) -> Self {
        args.pop_front()
            .expect("Missing variable from arg list")
            .as_const()
            .unwrap()
            .as_bool()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Variable> {
        [(*self).into()]
    }
}

impl FromArgList for u32 {
    fn from_arg_list(args: &mut VecDeque<Variable>) -> Self {
        args.pop_front()
            .expect("Missing variable from arg list")
            .as_const()
            .unwrap()
            .as_u32()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Variable> {
        [(*self).into()]
    }
}
