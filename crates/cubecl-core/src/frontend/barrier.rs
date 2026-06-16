//! This module exposes barrier for asynchronous data transfer

use alloc::vec;

use crate as cubecl;
use cubecl_ir::{
    ExpandValue,
    dialect::tma::*,
    pliron::{builtin::op_interfaces::OneResultInterface, context::Ptr, r#type::TypeObj},
    types::barrier::BarrierType,
};
use cubecl_macros::intrinsic;
use paste::paste;

use crate::{
    ir::{Scope, dialect::barrier::*},
    prelude::*,
    unexpanded,
};

use super::{CubePrimitive, CubeType, NativeExpand, SliceExpand, TensorMap};

/// A mechanism for awaiting on asynchronous data transfers
/// Behavior is defined by its ``BarrierLevel``.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Barrier;
pub type BarrierExpand = NativeExpand<Barrier>;

#[derive(Clone, Copy, PartialEq)]
pub struct BarrierToken;

impl CubeType for Barrier {
    type ExpandType = NativeExpand<Barrier>;
}

impl CubeDebug for Barrier {}

impl CubePrimitive for Barrier {
    type Scalar = u32; // Dummy, maybe we need another trait for non-standard primitives
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;
    fn from_const_value(_value: cubecl_ir::ConstantValue) -> Self {
        unreachable!("Can't create from const value")
    }

    fn __expand_as_type(scope: &Scope) -> Ptr<TypeObj> {
        BarrierType::get(scope.ctx()).into()
    }
}

impl NativeAssign for Barrier {
    fn elem_init_mut(_scope: &Scope, elem: ExpandValue) -> ExpandValue {
        elem
    }
}

impl CubeType for BarrierToken {
    type ExpandType = NativeExpand<BarrierToken>;
}

impl NativeAssign for BarrierToken {
    fn elem_init_mut(_scope: &Scope, elem: ExpandValue) -> ExpandValue {
        elem
    }
}

impl AsMutExpand for NativeExpand<BarrierToken> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

macro_rules! tensor_map_load {
    ($dim: literal, $($arg: expr),*) => {
        paste! {
            impl Barrier {
                /// Copy a tile from a global memory `source` to a shared memory `destination`, with
                /// the provided offsets.
                #[allow(unused, clippy::too_many_arguments)]
                pub fn [<tma_load_ $dim d>]<C1: CubePrimitive, C2: CubePrimitive<Scalar = C1::Scalar>>(
                    &self,
                    source: &TensorMap<C1, Tiled>,
                    destination: &mut [C2],
                    $($arg: i32),*
                ) {
                    unexpanded!()
                }

                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_ $dim d>]<C1: CubePrimitive, C2: CubePrimitive<Scalar = C1::Scalar>>(
                    scope: &Scope,
                    expand: &NativeExpand<Barrier>,
                    source: &NativeExpand<TensorMap<C1, Tiled>>,
                    destination: &mut SliceExpand<C2>,
                    $($arg: NativeExpand<i32>),*
                ) {
                    expand.[<__expand_tma_load_ $dim d_method>](scope, source, destination, $($arg),*);
                }
            }

            impl NativeExpand<Barrier> {
                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_ $dim d_method>]<C1: CubePrimitive, C2: CubePrimitive<Scalar = C1::Scalar>>(
                    &self,
                    scope: &Scope,
                    source: &NativeExpand<TensorMap<C1, Tiled>>,
                    destination: &mut SliceExpand<C2>,
                    $($arg: NativeExpand<i32>),*
                ) {
                    let barrier = self.value(scope);
                    let source = source.value(scope);
                    let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.value(scope);
                    let indices = vec![$($arg.read_value(scope)),*];

                    let mem_copy = TmaLoadOp::new(&mut scope.ctx_mut(), barrier, source, destination, indices);
                    scope.register(&mem_copy);
                }
            }
        }
    };
}

macro_rules! tensor_map_load_im2col {
    ($dim: literal, $($arg: expr),*; $($offset: expr),*) => {
        paste! {
            impl Barrier {
                /// Copy a tile from a global memory `source` to a shared memory `destination`, with
                /// the provided offsets.
                #[allow(unused, clippy::too_many_arguments)]
                pub fn [<tma_load_im2col_ $dim d>]<C1: CubePrimitive, C2: CubePrimitive<Scalar = C1::Scalar>>(
                    &self,
                    source: &TensorMap<C1, Im2col>,
                    destination: &mut [C2],
                    $($arg: i32,)*
                    $($offset: u16),*
                ) {
                    unexpanded!()
                }

                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_im2col_ $dim d>]<C1: CubePrimitive, C2: CubePrimitive<Scalar = C1::Scalar>>(
                    scope: &Scope,
                    expand: &NativeExpand<Barrier>,
                    source: &NativeExpand<TensorMap<C1, Im2col>>,
                    destination: &mut SliceExpand<C2>,
                    $($arg: NativeExpand<i32>,)*
                    $($offset: NativeExpand<u16>),*
                ) {
                    expand.[<__expand_tma_load_im2col_ $dim d_method>](scope, source, destination, $($arg),*, $($offset),*);
                }
            }

            impl NativeExpand<Barrier> {
                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_im2col_ $dim d_method>]<C1: CubePrimitive, C2: CubePrimitive<Scalar = C1::Scalar>>(
                    &self,
                    scope: &Scope,
                    source: &NativeExpand<TensorMap<C1, Im2col>>,
                    destination: &mut SliceExpand<C2>,
                    $($arg: NativeExpand<i32>,)*
                    $($offset: NativeExpand<u16>),*
                ) {
                    let barrier = self.value(scope);
                    let source = source.value(scope);
                    let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.value(scope);
                    let indices = vec![$($arg.read_value(scope)),*];
                    let offsets = vec![$($offset.read_value(scope)),*];

                    let mem_copy = TmaLoadIm2colOp::new(&mut scope.ctx_mut(), barrier, source, destination, indices, offsets);
                    scope.register(&mem_copy);
                }
            }
        }
    };
}

tensor_map_load!(1, x);
tensor_map_load!(2, y, x);
tensor_map_load!(3, z, y, x);
tensor_map_load!(4, w, z, y, x);
tensor_map_load!(5, v, w, z, y, x);

tensor_map_load_im2col!(3, n, w, c; w_offset);
tensor_map_load_im2col!(4, n, h, w, c; h_offset, w_offset);
tensor_map_load_im2col!(5, n, d, h, w, c; d_offset, h_offset, w_offset);

#[cube]
impl Barrier {
    /// Create a local barrier object for the current unit. Automatically initialized with an
    /// arrival count of `1`.
    pub fn local() -> Self {
        intrinsic!(|scope| {
            let value = scope.create_local_mut(BarrierType::get(&scope.ctx()));
            let arrival_count: ExpandValue = 1u32.into();
            let arrival_count = arrival_count.read_value(scope);
            let op = InitOp::new(&mut scope.ctx_mut(), value, arrival_count);
            scope.register(&op);
            value.into()
        })
    }

    /// Create a shared memory barrier that can be accesses by all units in the cube. Initialized
    /// by the `is_elected` unit with an arrival count of `arrival_count`. This is the number of
    /// times `arrive` or one of its variants needs to be called before the barrier advances.
    ///
    /// If all units in the cube arrive on the barrier, use `CUBE_DIM` as the arrival count. For
    /// other purposes, only a subset may need to arrive.
    pub fn shared(arrival_count: u32, is_elected: bool) -> Shared<Barrier> {
        intrinsic!(|scope| {
            let value = scope.create_shared(BarrierType::get(&scope.ctx()), None);
            if_expand(scope, is_elected, |scope| {
                let arrival_count = arrival_count.read_value(scope);
                let op = InitOp::new(&mut scope.ctx_mut(), value, arrival_count);
                scope.register(&op);
            });
            value.into()
        })
    }

    /// Create a shared memory barrier that can be accesses by all units in the cube. Only declared,
    /// but not initialized.
    pub fn shared_uninit() -> Shared<Barrier> {
        intrinsic!(|scope| {
            let value = scope.create_shared(BarrierType::get(&scope.ctx()), None);
            value.into()
        })
    }

    /// Initializes a barrier with a given `arrival_count`. This is the number of
    /// times `arrive` or one of its variants needs to be called before the barrier advances.
    ///
    /// If all units in the cube arrive on the barrier, use `CUBE_DIM` as the arrival count. For
    /// other purposes, only a subset may need to arrive.
    ///
    /// # Note
    ///
    /// No synchronization or election is performed, this is raw initialization. For shared barriers
    /// ensure only one unit performs the initialization, and synchronize the cube afterwards. There
    /// may also be additional synchronization requirements for bulk copy operations, like
    /// [`sync_async_proxy_shared()`].
    pub fn init_manual(&self, arrival_count: u32) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let arrival_count = arrival_count.read_value(scope);
            let op = InitOp::new(&mut scope.ctx_mut(), barrier, arrival_count);
            scope.register(&op);
        })
    }
}

// MemcpyAsync

#[cube]
impl Barrier {
    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async<C: CubePrimitive>(&self, source: &[C], destination: &mut [C]) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let source_length = source.__extract_length(scope).value(scope);
            let source = unsafe { *source.__expand_as_ptr_method(scope) }.value(scope);
            let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.value(scope);

            let mem_copy = MemCopyAsyncOp::new(
                &mut scope.ctx_mut(),
                barrier,
                destination,
                source,
                source_length,
                false.into(),
            );

            scope.register(&mem_copy);
        })
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async_cooperative<C: CubePrimitive>(&self, source: &[C], destination: &mut [C]) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let source_length = source.__extract_length(scope).value(scope);
            let source = unsafe { *source.__expand_as_ptr_method(scope) }.value(scope);
            let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.value(scope);

            let mem_copy = MemCopyAsyncOp::new(
                &mut scope.ctx_mut(),
                barrier,
                destination,
                source,
                source_length,
                true.into(),
            );

            scope.register(&mem_copy);
        })
    }

    /// Copy the source slice to destination. Uses transaction count like TMA, so use with
    /// `expect_tx` or `arrive_and_expect_tx`.
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async_tx<C: CubePrimitive>(&self, source: &[C], destination: &mut [C]) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let source_length = source.__extract_length(scope).value(scope);
            let source = unsafe { *source.__expand_as_ptr_method(scope) }.value(scope);
            let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.value(scope);

            let mem_copy = MemCopyAsyncTxOp::new(
                &mut scope.ctx_mut(),
                barrier,
                destination,
                source,
                source_length,
            );

            scope.register(&mem_copy);
        })
    }
}

// Arrival and Wait

#[cube]
impl Barrier {
    /// Arrive at the barrier, decrementing arrival count
    pub fn arrive(&self) -> BarrierToken {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let arrive = ArriveOp::new(&mut scope.ctx_mut(), barrier);
            scope.register(&arrive);
            arrive.get_result(&scope.ctx()).into()
        })
    }

    /// Arrive at the barrier, decrementing arrival count. Additionally increments expected count.
    pub fn arrive_and_expect_tx(&self, arrival_count: u32, transaction_count: u32) -> BarrierToken {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let arrival_count = arrival_count.read_value(scope);
            let transaction_count = transaction_count.read_value(scope);
            let op = ArriveAndExpectTxOp::new(
                &mut scope.ctx_mut(),
                barrier,
                arrival_count,
                transaction_count,
            );
            scope.register(&op);
            op.get_result(&scope.ctx()).into()
        })
    }

    /// Increments the expected count of the barrier.
    pub fn expect_tx(&self, transaction_count_update: u32) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let transaction_count_update = transaction_count_update.value(scope);
            scope.register(&ExpectTxOp::new(
                &mut scope.ctx_mut(),
                barrier,
                transaction_count_update,
            ));
        })
    }

    /// Wait until all data is loaded
    pub fn arrive_and_wait(&self) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            scope.register(&ArriveAndWaitOp::new(&mut scope.ctx_mut(), barrier));
        })
    }

    /// Wait at the barrier until all arrivals are done
    pub fn wait(&self, token: BarrierToken) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let token = token.value(scope);
            scope.register(&WaitOp::new(&mut scope.ctx_mut(), barrier, token));
        })
    }

    /// Wait at the barrier until the `phase` is completed. Doesn't require a token, but needs phase
    /// to be managed manually.
    pub fn wait_parity(&self, phase: u32) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            let phase = phase.read_value(scope);
            scope.register(&WaitParityOp::new(&mut scope.ctx_mut(), barrier, phase));
        })
    }
}

// Copy async

/// Copy the source slice in global memory to destination in shared memory with a low level async
/// copy. This only copies up to 128 bits/16 bytes, and does not synchronize. Use
/// `barrier.copy_async_arrive` to make the reads visible.
/// `copy_size` is in terms of elements to simplify copying between different vector sizes.
///
/// # Safety
///
/// This will try to copy the entire `copy_size`, so make sure the full width is in bounds.
/// Starting address must be aligned to the full copy size.
pub fn copy_async<C: CubePrimitive>(_source: &[C], _destination: &mut [C], _copy_size: u32) {
    unexpanded!()
}

pub mod copy_async {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        source: &SliceExpand<C>,
        destination: &mut SliceExpand<C>,
        copy_length: u32,
    ) {
        let source_length = ExpandValue::from(copy_length).read_value(scope);
        let source = unsafe { *source.__expand_as_ptr_method(scope) }.value(scope);
        let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.value(scope);
        let scalar_size = C::Scalar::__expand_size(scope);

        let mem_copy = CopyAsyncOp::new(
            scope.ctx_mut(),
            source,
            destination,
            source_length,
            (copy_length as usize * scalar_size).into(),
            false.into(),
        );

        scope.register(&mem_copy);
    }
}

/// Copy the source slice in global memory to destination in shared memory with a low level async
/// copy. This only copies up to 128 bits/16 bytes, and does not synchronize. Use
/// `barrier.copy_async_arrive` to make the reads visible.
/// `copy_size` is in terms of elements to simplify copying between different vector sizes.
///
/// Will only copy the length of the source slice, and zero fill the rest. Source length must be
/// <= copy size.
///
/// # Safety
/// Starting address must be aligned to the full copy size.
/// **This will silently fail if the address is only aligned to the source length and not the copy size!**
pub fn copy_async_checked<C: CubePrimitive>(
    _source: &[C],
    _destination: &mut [C],
    _copy_size: u32,
) {
    unexpanded!();
}

pub mod copy_async_checked {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        source: &SliceExpand<C>,
        destination: &mut SliceExpand<C>,
        copy_length: u32,
    ) {
        let source_length = source.__extract_length(scope).value(scope);

        // OOB pointer is allowed as long as length is 0
        let source = unsafe { *source.__expand_as_ptr_unchecked_method(scope) }.value(scope);
        let destination =
            unsafe { *destination.__expand_as_ptr_unchecked_method(scope) }.value(scope);
        let scalar_size = C::Scalar::__expand_size(scope);

        let mem_copy = CopyAsyncOp::new(
            scope.ctx_mut(),
            source,
            destination,
            source_length,
            (copy_length as usize * scalar_size).into(),
            true.into(),
        );

        scope.register(&mem_copy);
    }
}

#[cube]
impl Barrier {
    /// Makes all previous `copy_async` operations visible on the barrier.
    /// Should be called once after all copies have been dispatched, before reading from the shared
    /// memory.
    ///
    /// Does *not* count as an arrive in terms of the barrier arrival count. So `arrive` or
    /// `arrive_and_wait` should still be called afterwards.
    pub fn commit_copy_async(&self) {
        intrinsic!(|scope| {
            let barrier = self.value(scope);
            scope.register(&CommitCopyAsyncOp::new(&mut scope.ctx_mut(), barrier));
        })
    }
}

impl From<SharedExpand<Barrier>> for BarrierExpand {
    fn from(value: SharedExpand<Barrier>) -> Self {
        value.expand.into()
    }
}
