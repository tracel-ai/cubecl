use cubecl_ir::{AtomicOp, ExpandElement, StorageType};

use super::{
    ExpandElementIntoMut, ExpandElementTyped, Int, LaunchArgExpand, Numeric,
    into_mut_expand_element,
};
use crate::{
    frontend::{CubePrimitive, CubeType},
    ir::{BinaryOperator, CompareAndSwapOperator, Instruction, Scope, Type, UnaryOperator},
    prelude::KernelBuilder,
    unexpanded,
};

/// An atomic numerical type wrapping a normal numeric primitive. Enables the use of atomic
/// operations, while disabling normal operations. In WGSL, this is a separate type - on CUDA/SPIR-V
/// it can theoretically be bitcast to a normal number, but this isn't recommended.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Atomic<Inner: CubePrimitive> {
    pub val: Inner,
}

impl<Inner: Numeric> Atomic<Inner> {
    /// Load the value of the atomic.
    #[allow(unused_variables)]
    pub fn load(pointer: &Self) -> Inner {
        unexpanded!()
    }

    /// Store the value of the atomic.
    #[allow(unused_variables)]
    pub fn store(pointer: &Self, value: Inner) {
        unexpanded!()
    }

    /// Atomically stores the value into the atomic and returns the old value.
    #[allow(unused_variables)]
    pub fn swap(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn add(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    #[allow(unused_variables)]
    pub fn max(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    #[allow(unused_variables)]
    pub fn min(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn sub(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    pub fn __expand_load(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let pointer: ExpandElement = pointer.into();
        let new_var = scope.create_local(Type::new(Inner::as_type(scope)));
        scope.register(Instruction::new(
            AtomicOp::Load(UnaryOperator { input: *pointer }),
            *new_var,
        ));
        new_var.into()
    }

    pub fn __expand_store(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        scope.register(Instruction::new(
            AtomicOp::Store(UnaryOperator { input: *value }),
            *ptr,
        ));
    }

    pub fn __expand_swap(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }

    pub fn __expand_add(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }

    pub fn __expand_sub(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }

    pub fn __expand_max(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }

    pub fn __expand_min(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }
}

impl<Inner: Int> Atomic<Inner> {
    /// Compare the value at `pointer` to `cmp` and set it to `value` only if they are the same.
    /// Returns the old value of the pointer before the store.
    ///
    /// ### Tip
    /// Compare the returned value to `cmp` to determine whether the store was successful.
    #[allow(unused_variables)]
    pub fn compare_and_swap(pointer: &Self, cmp: Inner, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn and(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn or(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    pub fn xor(pointer: &Self, value: Inner) -> Inner {
        unexpanded!()
    }

    pub fn __expand_compare_and_swap(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        cmp: <Inner as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let pointer: ExpandElement = pointer.into();
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
    }

    pub fn __expand_and(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }

    pub fn __expand_or(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
    }

    pub fn __expand_xor(
        scope: &mut Scope,
        pointer: <Self as CubeType>::ExpandType,
        value: <Inner as CubeType>::ExpandType,
    ) -> <Inner as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
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
}

impl<Inner: CubePrimitive> ExpandElementIntoMut for Atomic<Inner> {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl<Inner: CubePrimitive> LaunchArgExpand for Atomic<Inner> {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(Self::as_type_native_unchecked()).into()
    }
}
