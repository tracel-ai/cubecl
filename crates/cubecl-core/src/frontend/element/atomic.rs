use cubecl_ir::{
    ConstantValue, ExpandValue,
    dialect::{atomic::*, synchronization::SyncScope},
    types::AtomicType,
};
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
    fn __expand_fetch_add(
        scope: &Scope,
        ptr: ExpandValue,
        value: ExpandValue,
        sync: SyncScope,
    ) -> ExpandValue;
    fn __expand_fetch_sub(
        scope: &Scope,
        ptr: ExpandValue,
        value: ExpandValue,
        sync: SyncScope,
    ) -> ExpandValue;
    fn __expand_fetch_min(
        scope: &Scope,
        ptr: ExpandValue,
        value: ExpandValue,
        sync: SyncScope,
    ) -> ExpandValue;
    fn __expand_fetch_max(
        scope: &Scope,
        ptr: ExpandValue,
        value: ExpandValue,
        sync: SyncScope,
    ) -> ExpandValue;
}

macro_rules! atomic_numeric {
    ($($ty: ty),*; $add: ty, $sub: ty, $min: ty, $max: ty) => {
        $(impl AtomicNumeric for $ty {
            fn __expand_fetch_add(scope: &Scope, ptr: ExpandValue, value: ExpandValue, sync: SyncScope) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, sync, <$add>::new)
            }
            fn __expand_fetch_sub(scope: &Scope, ptr: ExpandValue, value: ExpandValue, sync: SyncScope) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, sync, <$sub>::new)
            }
            fn __expand_fetch_min(scope: &Scope, ptr: ExpandValue, value: ExpandValue, sync: SyncScope) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, sync, <$min>::new)
            }
            fn __expand_fetch_max(scope: &Scope, ptr: ExpandValue, value: ExpandValue, sync: SyncScope) -> ExpandValue {
                atomic_binary_expand(scope, ptr, value, sync, <$max>::new)
            }
        })*
    };
}

atomic_numeric!(i8, i16, i32, i64, isize; AtomicIAddOp, AtomicISubOp, AtomicSMinOp, AtomicSMaxOp);
atomic_numeric!(u8, u16, u32, u64, usize; AtomicIAddOp, AtomicISubOp, AtomicUMinOp, AtomicUMaxOp);
atomic_numeric!(f16, bf16, f32, flex32, tf32, f64; AtomicFAddOp, AtomicFSubOp, AtomicFMinOp, AtomicFMaxOp);

/// The scope an atomic on `ptr` synchronizes at: `sync`, or the system's when `ptr` points into a
/// [`Peer`] buffer, which kernels on other devices reach while this one runs.
fn reach(scope: &Scope, ptr: Value, sync: SyncScope) -> SyncScope {
    match scope.is_peer_pointer(ptr) {
        true => sync.max(SyncScope::System),
        false => sync,
    }
}

fn atomic_binary_expand<F, O>(
    scope: &Scope,
    ptr: ExpandValue,
    value: ExpandValue,
    sync: SyncScope,
    func: F,
) -> ExpandValue
where
    F: Fn(&mut Context, Value, Value) -> O,
    O: Op + OneResultInterface,
{
    let ptr = ptr.value(scope);
    let sync = reach(scope, ptr, sync);
    let op = func(scope.ctx_mut(), ptr, value.read_value(scope));
    set_atomic_scope(scope.ctx_mut(), op.get_operation(), sync);
    scope.register_with_result(&op).into()
}

#[cube]
impl<Inner: CubePrimitive<Scalar: AtomicNumeric>> Atomic<Inner> {
    /// Load the value of the atomic.
    pub fn load(&self) -> Inner {
        self.load_at(comptime![SyncScope::Device])
    }

    /// Store the value of the atomic.
    pub fn store(&self, value: Inner) {
        self.store_at(value, comptime![SyncScope::Device])
    }

    /// Atomically stores the value into the atomic and returns the old value.
    pub fn exchange(&self, value: Inner) -> Inner {
        self.exchange_at(value, comptime![SyncScope::Device])
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    pub fn fetch_add(&self, value: Inner) -> Inner {
        self.fetch_add_at(value, comptime![SyncScope::Device])
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    pub fn fetch_sub(&self, value: Inner) -> Inner {
        self.fetch_sub_at(value, comptime![SyncScope::Device])
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    pub fn fetch_max(&self, value: Inner) -> Inner {
        self.fetch_max_at(value, comptime![SyncScope::Device])
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    pub fn fetch_min(&self, value: Inner) -> Inner {
        self.fetch_min_at(value, comptime![SyncScope::Device])
    }

    /// `load`, synchronizing at `sync`.
    ///
    /// The plain operations synchronize at `SyncScope::Device`, and at
    /// `SyncScope::System` on a `Peer` buffer, whatever scope they name. The system scope
    /// also synchronizes with the host and with kernels on other devices that bind this memory,
    /// which is what kernels on different devices signal each other through: publish with
    /// `sync_system()`, then store a flag; spin on the flag, then
    /// `sync_system()` before reading what it guards.
    #[allow(unused_variables)]
    pub fn load_at(&self, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let op = AtomicLoadOp::new(scope.ctx_mut(), ptr);
            let sync = reach(scope, ptr, sync);
            set_atomic_scope(scope.ctx_mut(), op.get_operation(), sync);
            scope.register_with_result(&op).into()
        })
    }

    /// `store`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn store_at(&self, value: Inner, #[comptime] sync: SyncScope) {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicStoreOp::new(scope.ctx_mut(), ptr, value);
            let sync = reach(scope, ptr, sync);
            set_atomic_scope(scope.ctx_mut(), op.get_operation(), sync);
            scope.register(&op);
        })
    }

    /// `exchange`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn exchange_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let value = value.read_value(scope);
            let op = AtomicExchangeOp::new(scope.ctx_mut(), ptr, value);
            let sync = reach(scope, ptr, sync);
            set_atomic_scope(scope.ctx_mut(), op.get_operation(), sync);
            scope.register_with_result(&op).into()
        })
    }

    /// `fetch_add`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_add_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| Inner::Scalar::__expand_fetch_add(
            scope,
            self.expand,
            value.expand,
            sync
        )
        .into())
    }

    /// `fetch_sub`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_sub_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| Inner::Scalar::__expand_fetch_sub(
            scope,
            self.expand,
            value.expand,
            sync
        )
        .into())
    }

    /// `fetch_max`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_max_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| Inner::Scalar::__expand_fetch_max(
            scope,
            self.expand,
            value.expand,
            sync
        )
        .into())
    }

    /// `fetch_min`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_min_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| Inner::Scalar::__expand_fetch_min(
            scope,
            self.expand,
            value.expand,
            sync
        )
        .into())
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
        self.compare_exchange_weak_at(cmp, value, comptime![SyncScope::Device])
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    pub fn fetch_and(&self, value: Inner) -> Inner {
        self.fetch_and_at(value, comptime![SyncScope::Device])
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    pub fn fetch_or(&self, value: Inner) -> Inner {
        self.fetch_or_at(value, comptime![SyncScope::Device])
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    pub fn fetch_xor(&self, value: Inner) -> Inner {
        self.fetch_xor_at(value, comptime![SyncScope::Device])
    }

    /// `compare_exchange_weak`, synchronizing at `sync` (see
    /// `load_at`).
    #[allow(unused_variables)]
    pub fn compare_exchange_weak_at(
        &self,
        cmp: Inner,
        value: Inner,
        #[comptime] sync: SyncScope,
    ) -> Inner {
        intrinsic!(|scope| {
            let ptr = self.value(scope);
            let cmp = cmp.read_value(scope);
            let value = value.read_value(scope);
            let op = AtomicCompareExchangeWeakOp::new(scope.ctx_mut(), ptr, cmp, value);
            let sync = reach(scope, ptr, sync);
            set_atomic_scope(scope.ctx_mut(), op.get_operation(), sync);
            scope.register_with_result(&op).into()
        })
    }

    /// `fetch_and`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_and_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| atomic_binary_expand(
            scope,
            self.expand,
            value.expand,
            sync,
            AtomicAndOp::new
        )
        .into())
    }

    /// `fetch_or`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_or_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| atomic_binary_expand(
            scope,
            self.expand,
            value.expand,
            sync,
            AtomicOrOp::new
        )
        .into())
    }

    /// `fetch_xor`, synchronizing at `sync` (see `load_at`).
    #[allow(unused_variables)]
    pub fn fetch_xor_at(&self, value: Inner, #[comptime] sync: SyncScope) -> Inner {
        intrinsic!(|scope| atomic_binary_expand(
            scope,
            self.expand,
            value.expand,
            sync,
            AtomicXorOp::new
        )
        .into())
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

    fn from_expand_elem(elem: ExpandValue) -> Self::ExpandType {
        NativeExpand::new(elem)
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        panic!("Can't have constant atomic");
    }
}
impl<Inner: CubePrimitive> ConstantTyped for Atomic<Inner> {
    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        let inner = Inner::__expand_as_type(scope);
        AtomicType::get(scope.ctx(), inner).into()
    }
}

impl<Inner: CubePrimitive> NativeAssign for Atomic<Inner> {}
