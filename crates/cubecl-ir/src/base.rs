use std::collections::VecDeque;

use smallvec::SmallVec;

use crate::Variable;

pub trait OperationCore: Sized {
    type OpCode;

    fn op_code(&self) -> Self::OpCode;
    fn args(&self) -> Option<SmallVec<[Variable; 4]>> {
        None
    }
    #[allow(unused)]
    fn from_code_and_args(op_code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        None
    }
}

pub trait OperationArgs: Sized {
    #[allow(unused)]
    fn from_args(args: &[Variable]) -> Option<Self> {
        None
    }

    fn as_args(&self) -> Option<SmallVec<[Variable; 4]>> {
        None
    }
}

pub trait FromArgList: Sized {
    fn from_arg_list(args: &mut VecDeque<Variable>) -> Self;
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
        self.clone()
    }
}
