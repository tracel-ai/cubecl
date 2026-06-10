use alloc::collections::VecDeque;

use alloc::vec;
use alloc::vec::Vec;

use crate::{Builtin, Instruction, Memory, Scope, Type, Value};

/// An operation that can be reflected on
pub trait OperationReflect: Sized {
    /// Type of the op codes for this operation
    type OpCode;

    /// Get the opcode for this operation
    fn op_code(&self) -> Self::OpCode;
    /// Get the list of arguments for this operation. If not all arguments are [`Value`], returns
    /// `None` instead.
    fn args(&self) -> Option<Vec<Value>> {
        None
    }
    fn args_mut(&mut self) -> Option<Vec<&mut Value>> {
        None
    }
    /// Create typed operation from an opcode and a list of arguments. Returns `None` if not all
    /// arguments are [`Value`].
    #[allow(unused)]
    fn from_code_and_args(op_code: Self::OpCode, args: &[Value]) -> Option<Self> {
        None
    }
    /// Sanitize args, i.e. loading inputs from pointers for ops that take values
    fn sanitize_args(&mut self, scope: &Scope);
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

    fn read_pointers(&self) -> Vec<Value> {
        Vec::new()
    }

    fn write_pointers(&self) -> Vec<Value> {
        Vec::new()
    }
}

/// A type that represents an operation's arguments
pub trait OperationArgs: Sized {
    /// Sanitize args for the `ptr` constraint, loading inputs from pointers for ops that take values.
    /// This is needed because Rust "helpfully" auto-derefs references, skipping the explicit deref
    /// code that inserts a [`Memory::Load`].
    fn sanitize_args_ptr(&mut self, scope: &Scope);

    /// Construct this type from a list of arguments. If not all arguments are [`Value`], returns
    /// `None`
    #[allow(unused)]
    fn from_args(args: &[Value]) -> Option<Self> {
        None
    }

    /// Turns this type into a flat list of arguments. If not all arguments are [`Value`],
    /// returns `None`
    fn as_args(&self) -> Option<Vec<Value>> {
        None
    }

    /// Turns this type into a flat list of arguments. If not all arguments are [`Value`],
    /// returns `None`
    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        None
    }

    fn read_pointers(&self) -> Vec<Value> {
        Vec::new()
    }

    fn write_pointers(&self) -> Vec<Value> {
        Vec::new()
    }
}

impl OperationArgs for Value {
    fn sanitize_args_ptr(&mut self, scope: &Scope) {
        *self = read_value(scope, *self)
    }

    fn from_args(args: &[Value]) -> Option<Self> {
        Some(args[0])
    }

    fn as_args(&self) -> Option<Vec<Value>> {
        Some(vec![*self])
    }

    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        Some(vec![self])
    }
}

impl<T: OperationArgs> OperationArgs for Vec<T> {
    fn sanitize_args_ptr(&mut self, scope: &Scope) {
        self.iter_mut().for_each(|it| it.sanitize_args_ptr(scope));
    }

    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        let inner = self.iter_mut().map(|it| it.as_args_mut());
        let inner = inner.collect::<Option<Vec<_>>>()?;
        Some(inner.into_iter().flatten().collect())
    }
}

impl<T: OperationArgs> OperationArgs for Option<T> {
    fn sanitize_args_ptr(&mut self, scope: &Scope) {
        if let Some(it) = self.as_mut() {
            it.sanitize_args_ptr(scope)
        }
    }

    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        self.as_mut().and_then(|inner| inner.as_args_mut())
    }
}

impl OperationArgs for usize {
    fn sanitize_args_ptr(&mut self, _: &Scope) {}

    fn from_args(args: &[Value]) -> Option<Self> {
        Some(args[0].as_const().unwrap().as_usize())
    }
    fn as_args(&self) -> Option<Vec<Value>> {
        Some(vec![(*self).into()])
    }
    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        Some(vec![])
    }
}
impl OperationArgs for u32 {
    fn sanitize_args_ptr(&mut self, _: &Scope) {}

    fn from_args(args: &[Value]) -> Option<Self> {
        Some(args[0].as_const().unwrap().as_u32())
    }
    fn as_args(&self) -> Option<Vec<Value>> {
        Some(vec![(*self).into()])
    }
    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        Some(vec![])
    }
}
impl OperationArgs for bool {
    fn sanitize_args_ptr(&mut self, _: &Scope) {}

    fn from_args(args: &[Value]) -> Option<Self> {
        Some(args[0].as_const().unwrap().as_bool())
    }
    fn as_args(&self) -> Option<Vec<Value>> {
        Some(vec![(*self).into()])
    }
    fn as_args_mut(&mut self) -> Option<Vec<&mut Value>> {
        Some(vec![])
    }
}

/// Types that can be destructured into and created from a list of [`Value`]s.
pub trait FromArgList: Sized {
    /// Creates this type from a list of values. This works like a parse stream, where consumed
    /// values are popped from the front.
    fn from_arg_list(args: &mut VecDeque<Value>) -> Self;
    /// Turns this type into a list of [`Value`]s.
    fn as_arg_list(&self) -> impl IntoIterator<Item = Value>;
    fn as_arg_list_mut(&mut self) -> impl IntoIterator<Item = &mut Value>;
}

impl FromArgList for Value {
    fn from_arg_list(args: &mut VecDeque<Value>) -> Self {
        args.pop_front().expect("Missing value from arg list")
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Value> {
        [*self]
    }

    fn as_arg_list_mut(&mut self) -> impl IntoIterator<Item = &mut Value> {
        [self]
    }
}

impl FromArgList for Vec<Value> {
    fn from_arg_list(args: &mut VecDeque<Value>) -> Self {
        core::mem::take(args).into_iter().collect()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Value> {
        self.iter().cloned()
    }

    fn as_arg_list_mut(&mut self) -> impl IntoIterator<Item = &mut Value> {
        self.iter_mut()
    }
}

impl FromArgList for bool {
    fn from_arg_list(args: &mut VecDeque<Value>) -> Self {
        args.pop_front()
            .expect("Missing value from arg list")
            .as_const()
            .unwrap()
            .as_bool()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Value> {
        [(*self).into()]
    }

    fn as_arg_list_mut(&mut self) -> impl IntoIterator<Item = &mut Value> {
        []
    }
}

impl FromArgList for u32 {
    fn from_arg_list(args: &mut VecDeque<Value>) -> Self {
        args.pop_front()
            .expect("Missing value from arg list")
            .as_const()
            .unwrap()
            .as_u32()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Value> {
        [(*self).into()]
    }

    fn as_arg_list_mut(&mut self) -> impl IntoIterator<Item = &mut Value> {
        []
    }
}

impl OperationArgs for Builtin {
    fn sanitize_args_ptr(&mut self, _: &crate::Scope) {}
}

impl FromArgList for usize {
    fn from_arg_list(args: &mut VecDeque<Value>) -> Self {
        args.pop_front()
            .expect("Missing value from arg list")
            .as_const()
            .unwrap()
            .as_usize()
    }

    fn as_arg_list(&self) -> impl IntoIterator<Item = Value> {
        [(*self).into()]
    }

    fn as_arg_list_mut(&mut self) -> impl IntoIterator<Item = &mut Value> {
        []
    }
}

fn read_value(scope: &Scope, val: Value) -> Value {
    if let Type::Pointer(inner, _) = val.ty {
        let out = scope.create_value(*inner);
        scope.register(Instruction::new(Memory::Load(val), out));
        out
    } else {
        val
    }
}
