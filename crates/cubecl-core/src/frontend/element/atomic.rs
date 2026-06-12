use cubecl_ir::{
    ConstantValue, ExpandValue,
    dialect::atomic::{
        AtomicAddOp, AtomicAndOp, AtomicCompareExchangeWeakOp, AtomicExchangeOp, AtomicLoadOp,
        AtomicMaxOp, AtomicMinOp, AtomicOrOp, AtomicStoreOp, AtomicSubOp, AtomicXorOp,
    },
    pliron::{builtin::op_interfaces::OneResultInterface, context::Ptr, r#type::TypeObj},
    types::AtomicType,
};
use cubecl_macros::intrinsic;

use super::{NativeAssign, NativeExpand, Numeric};
use crate::{
    self as cubecl,
    frontend::{CubePrimitive, CubeType},
    ir::Scope,
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
    pub fn load(&self) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let op = AtomicLoadOp::new(&mut scope.ctx_mut(), ptr);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Store the value of the atomic.
    pub fn store(&self, value: Inner) {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            scope.register(&AtomicStoreOp::new(&mut scope.ctx_mut(), ptr, value));
        })
    }

    /// Atomically stores the value into the atomic and returns the old value.
    pub fn exchange(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicExchangeOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    pub fn fetch_add(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicAddOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    pub fn fetch_sub(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicSubOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    pub fn fetch_max(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicMaxOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    pub fn fetch_min(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicMinOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
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
    pub fn compare_exchange_weak(&self, cmp: Inner, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let cmp = cmp.read_value(scope);
            let value = value.read_value(scope);
            let op = AtomicCompareExchangeWeakOp::new(&mut scope.ctx_mut(), ptr, cmp, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    pub fn fetch_and(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicAndOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    pub fn fetch_or(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicOrOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    pub fn fetch_xor(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicXorOp::new(&mut scope.ctx_mut(), ptr, value);
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
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

    fn __expand_as_type(scope: &Scope) -> Ptr<TypeObj> {
        let inner = Inner::__expand_as_type(scope);
        AtomicType::get(&mut scope.ctx_mut(), inner).into()
    }

    fn from_expand_elem(elem: ExpandValue) -> Self::ExpandType {
        NativeExpand::new(elem)
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        panic!("Can't have constant atomic");
    }
}

impl<Inner: CubePrimitive> NativeAssign for Atomic<Inner> {}
