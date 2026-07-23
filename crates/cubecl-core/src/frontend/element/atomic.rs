use cubecl_ir::{ConstantValue, ExpandValue, dialect::atomic::*, types::AtomicType};
use cubecl_macros::intrinsic;
use half::{bf16, f16};
use pliron::{
    builtin::op_interfaces::OneResultInterface, context::Context, op::Op, r#type::TypeHandle,
    value::Value,
};

use super::{NativeAssign, NativeExpand};
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

pub trait AtomicNumeric {
    fn __expand_fetch_add(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue;
    fn __expand_fetch_sub(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue;
    fn __expand_fetch_min(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue;
    fn __expand_fetch_max(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue;
}

macro_rules! atomic_numeric {
    ($($ty: ty),*; $add: ty, $sub: ty, $min: ty, $max: ty) => {
        $(impl AtomicNumeric for $ty {
            fn __expand_fetch_add(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, <$add>::new)
            }
            fn __expand_fetch_sub(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, <$sub>::new)
            }
            fn __expand_fetch_min(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, <$min>::new)
            }
            fn __expand_fetch_max(scope: &Scope, ptr: ExpandValue, value: ExpandValue) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, <$max>::new)
            }
        })*
    };
}

atomic_numeric!(i8, i16, i32, i64, isize; AtomicIAddOp, AtomicISubOp, AtomicSMinOp, AtomicSMaxOp);
atomic_numeric!(u8, u16, u32, u64, usize; AtomicIAddOp, AtomicISubOp, AtomicUMinOp, AtomicUMaxOp);
atomic_numeric!(f16, bf16, f32, flex32, tf32, f64; AtomicFAddOp, AtomicFSubOp, AtomicFMinOp, AtomicFMaxOp);

fn atomic_binary_expand<F, O>(
    scope: &Scope,
    ptr: ExpandValue,
    value: ExpandValue,
    func: F,
) -> ExpandValue
where
    F: Fn(&mut Context, Value, Value) -> O,
    O: Op + OneResultInterface,
{
    let op = func(scope.ctx_mut(), ptr.value(scope), value.read_value(scope));
    scope.register_with_result(&op).into()
}

#[cube]
impl<Inner: CubePrimitive<Scalar: AtomicNumeric>> Atomic<Inner> {
    /// Load the value of the atomic.
    pub fn load(&self) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let op = AtomicLoadOp::new(scope.ctx_mut(), ptr);
            scope.register_with_result(&op).into()
        })
    }

    /// Store the value of the atomic.
    pub fn store(&self, value: Inner) {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            scope.register(&AtomicStoreOp::new(scope.ctx_mut(), ptr, value));
        })
    }

    /// Atomically stores the value into the atomic and returns the old value.
    pub fn exchange(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicExchangeOp::new(scope.ctx_mut(), ptr, value);
            scope.register_with_result(&op).into()
        })
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    pub fn fetch_add(&self, value: Inner) -> Inner {
        intrinsic!(
            |scope| Inner::Scalar::__expand_fetch_add(scope, self.expand, value.expand).into()
        )
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    pub fn fetch_sub(&self, value: Inner) -> Inner {
        intrinsic!(
            |scope| Inner::Scalar::__expand_fetch_sub(scope, self.expand, value.expand).into()
        )
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    pub fn fetch_max(&self, value: Inner) -> Inner {
        intrinsic!(
            |scope| Inner::Scalar::__expand_fetch_max(scope, self.expand, value.expand).into()
        )
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    pub fn fetch_min(&self, value: Inner) -> Inner {
        intrinsic!(
            |scope| Inner::Scalar::__expand_fetch_min(scope, self.expand, value.expand).into()
        )
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
            let op = AtomicCompareExchangeWeakOp::new(scope.ctx_mut(), ptr, cmp, value);
            scope.register_with_result(&op).into()
        })
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    pub fn fetch_and(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicAndOp::new(scope.ctx_mut(), ptr, value);
            scope.register_with_result(&op).into()
        })
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    pub fn fetch_or(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicOrOp::new(scope.ctx_mut(), ptr, value);
            scope.register_with_result(&op).into()
        })
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    pub fn fetch_xor(&self, value: Inner) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicXorOp::new(scope.ctx_mut(), ptr, value);
            scope.register_with_result(&op).into()
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

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        let inner = Inner::__expand_as_type(scope);
        AtomicType::get(scope.ctx(), inner).into()
    }

    fn from_expand_elem(elem: ExpandValue) -> Self::ExpandType {
        NativeExpand::new(elem)
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        panic!("Can't have constant atomic");
    }
}

impl<Inner: CubePrimitive> NativeAssign for Atomic<Inner> {}
