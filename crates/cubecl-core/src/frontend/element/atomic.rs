use cubecl_ir::{AtomicBinaryOperands, AtomicOp, ConstantValue, StoreOperands, Variable};
use cubecl_macros::intrinsic;

use super::{NativeAssign, NativeExpand, Numeric};
use crate::{
    self as cubecl,
    frontend::{CubePrimitive, CubeType},
    ir::{CompareAndSwapOperands, Instruction, Scope, Type},
    prelude::*,
};

/// An atomic numerical type wrapping a normal numeric primitive. Enables the use of atomic
/// operations, while disabling normal operations. In WGSL, this is a separate type - on CUDA/SPIR-V
/// it can theoretically be bitcast to a normal number, but this isn't recommended.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Atomic<Inner: CubePrimitive> {
    pub val: Inner,
}

type AtomicExpand<Inner> = NativeExpand<Atomic<Inner>>;

#[cube]
impl<Inner: CubePrimitive<Scalar: Numeric>> Atomic<Inner> {
    /// Load the value of the atomic.
    #[allow(unused_variables)]
    pub fn load(&self) -> Inner {
        intrinsic!(|scope| {
            let pointer: Variable = self.clone().into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(AtomicOp::Load(pointer), new_var));
            new_var.into()
        })
    }

    /// Store the value of the atomic.
    #[allow(unused_variables)]
    pub fn store(&self, value: Inner) {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            scope.register(Instruction::no_out(AtomicOp::Store(StoreOperands {
                ptr,
                value,
            })));
        })
    }

    /// Atomically stores the value into the atomic and returns the old value.
    #[allow(unused_variables)]
    pub fn swap(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Swap(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn fetch_add(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Add(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn fetch_sub(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Sub(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    #[allow(unused_variables)]
    pub fn fetch_max(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Max(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    #[allow(unused_variables)]
    pub fn fetch_min(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Min(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }
}

#[cube]
impl<Inner: CubePrimitive<Scalar: Int>> Atomic<Inner> {
    /// Compare the value at `pointer` to `cmp` and set it to `value` only if they are the same.
    /// Returns the old value of the pointer before the store.
    ///
    /// ### Tip
    /// Compare the returned value to `cmp` to determine whether the store was successful.
    #[allow(unused_variables)]
    pub fn compare_exchange_weak(&self, cmp: Inner, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let pointer: Variable = self.clone().into();
            let cmp: Variable = cmp.into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::CompareAndSwap(CompareAndSwapOperands {
                    ptr: pointer,
                    cmp: cmp,
                    val: value,
                }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn fetch_and(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::And(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn fetch_or(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Or(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn fetch_xor(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: Variable = self.clone().into();
            let value: Variable = value.into();
            let new_var = scope.create_local(Inner::__expand_as_type(scope));
            scope.register(Instruction::new(
                AtomicOp::Xor(AtomicBinaryOperands { ptr, value }),
                new_var,
            ));
            new_var.into()
        })
    }
}

impl<Inner: CubePrimitive> CubeType for Atomic<Inner> {
    type ExpandType = NativeExpand<Self>;
}

impl<Inner: CubePrimitive> CubeDebug for Atomic<Inner> {}
impl<Inner: CubePrimitive> CubePrimitive for Atomic<Inner> {
    type Scalar = Inner::Scalar;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = Atomic<S>;

    fn as_type_native() -> Option<Type> {
        Inner::as_type_native().map(Type::atomic)
    }

    fn __expand_as_type(scope: &Scope) -> Type {
        Type::atomic(Inner::__expand_as_type(scope))
    }

    fn as_type_native_unchecked() -> Type {
        Type::atomic(Inner::as_type_native_unchecked())
    }

    fn size() -> Option<usize> {
        Inner::size()
    }

    fn from_expand_elem(elem: Variable) -> Self::ExpandType {
        NativeExpand::new(elem)
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        panic!("Can't have constant atomic");
    }
}

impl<Inner: CubePrimitive> NativeAssign for Atomic<Inner> {}
