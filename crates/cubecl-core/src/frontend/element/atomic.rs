use cubecl_ir::{AtomicOp, ConstantValue, ExpandElement, StorageType};
use cubecl_macros::intrinsic;

use super::{ExpandElementIntoMut, ExpandElementTyped, Int, Numeric, into_mut_expand_element};
use crate::{
    self as cubecl,
    frontend::{CubePrimitive, CubeType},
    ir::{BinaryOperator, CompareAndSwapOperator, Instruction, Scope, Type, UnaryOperator},
    prelude::*,
};

/// An atomic numerical type wrapping a normal numeric primitive. Enables the use of atomic
/// operations, while disabling normal operations. In WGSL, this is a separate type - on CUDA/SPIR-V
/// it can theoretically be bitcast to a normal number, but this isn't recommended.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Atomic<Inner: CubePrimitive> {
    pub val: Inner,
}

type AtomicExpand<Inner> = ExpandElementTyped<Atomic<Inner>>;

#[cube]
impl<Inner: Numeric> Atomic<Inner> {
    /// Load the value of the atomic.
    #[allow(unused_variables)]
    pub fn load(&self) -> Inner {
        intrinsic!(|scope| {
            let pointer: ExpandElement = self.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Load(UnaryOperator { input: *pointer }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Store the value of the atomic.
    #[allow(unused_variables)]
    pub fn store(&self, value: Inner) {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            scope.register(Instruction::new(
                AtomicOp::Store(UnaryOperator { input: *value }),
                *ptr,
            ));
        })
    }

    /// Atomically stores the value into the atomic and returns the old value.
    #[allow(unused_variables)]
    pub fn swap(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Swap(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn add(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Add(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn sub(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Sub(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    #[allow(unused_variables)]
    pub fn max(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Max(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    #[allow(unused_variables)]
    pub fn min(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Min(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }
}

#[cube]
impl<Inner: Int> Atomic<Inner> {
    /// Compare the value at `pointer` to `cmp` and set it to `value` only if they are the same.
    /// Returns the old value of the pointer before the store.
    ///
    /// ### Tip
    /// Compare the returned value to `cmp` to determine whether the store was successful.
    #[allow(unused_variables)]
    pub fn compare_and_swap(&self, cmp: Inner, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let pointer: ExpandElement = self.into();
            let cmp: ExpandElement = cmp.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::CompareAndSwap(CompareAndSwapOperator {
                    input: *pointer,
                    cmp: *cmp,
                    val: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn and(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::And(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn or(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Or(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn xor(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr: ExpandElement = self.into();
            let value: ExpandElement = value.into();
            let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
            scope.register(Instruction::new(
                AtomicOp::Xor(BinaryOperator {
                    lhs: *ptr,
                    rhs: *value,
                }),
                *new_var,
            ));
            new_var.into()
        })
    }
}

impl<Inner: CubePrimitive> CubeType for Atomic<Inner> {
    type ExpandType = ExpandElementTyped<Self>;
}

impl<Inner: CubePrimitive> CubePrimitive for Atomic<Inner> {
    fn as_type_native() -> Option<StorageType> {
        Inner::as_type_native().map(|it| StorageType::Atomic(it.elem_type()))
    }

    fn as_type(scope: &Scope) -> StorageType {
        StorageType::Atomic(Inner::as_type(scope).elem_type())
    }

    fn as_type_native_unchecked() -> StorageType {
        StorageType::Atomic(Inner::as_type_native_unchecked().elem_type())
    }

    fn size() -> Option<usize> {
        Inner::size()
    }

    fn from_expand_elem(elem: ExpandElement) -> Self::ExpandType {
        ExpandElementTyped::new(elem)
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        panic!("Can't have constant atomic");
    }
}

impl<Inner: CubePrimitive> ExpandElementIntoMut for Atomic<Inner> {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}
