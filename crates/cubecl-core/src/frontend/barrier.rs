//! This module exposes barrier for asynchronous data transfer

use alloc::vec;

use crate as cubecl;
use cubecl_ir::{Instruction, OpaqueType, Variable};
use cubecl_macros::intrinsic;
use paste::paste;

use crate::{
    ir::{BarrierOps, Scope},
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
}

impl NativeAssign for Barrier {
    fn elem_init_mut(_scope: &Scope, elem: Variable) -> Variable {
        elem
    }
}

impl CubeType for BarrierToken {
    type ExpandType = NativeExpand<BarrierToken>;
}

impl NativeAssign for BarrierToken {
    fn elem_init_mut(_scope: &Scope, elem: Variable) -> Variable {
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
                    let barrier = self.expand;
                    let source = source.expand;
                    let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.expand;

                    let mem_copy = BarrierOps::TmaLoad {
                        barrier,
                        tensor_map: source,
                        destination,
                        indices: vec![$($arg.expand),*],
                    };

                    scope.register(Instruction::no_out(mem_copy));
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
                    let barrier = self.expand;
                    let source = source.expand;
                    let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.expand;

                    let mem_copy = BarrierOps::TmaLoadIm2col {
                        barrier,
                        tensor_map: source,
                        destination,
                        indices: vec![$($arg.expand),*],
                        offsets: vec![$($offset.expand),*],
                    };

                    scope.register(Instruction::no_out(mem_copy));
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
            let variable =
                scope.create_local_mut(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Unit));
            scope.register(BarrierOps::Init {
                barrier: variable,
                is_elected: true.into(),
                arrival_count: 1.into(),
            });
            variable.into()
        })
    }

    /// Create a shared memory barrier that can be accesses by all units in the cube. Initialized
    /// by the `is_elected` unit with an arrival count of `arrival_count`. This is the number of
    /// times `arrive` or one of its variants needs to be called before the barrier advances.
    ///
    /// If all units in the cube arrive on the barrier, use `CUBE_DIM` as the arrival count. For
    /// other purposes, only a subset may need to arrive.
    #[allow(unused_variables)]
    pub fn shared(arrival_count: u32, is_elected: bool) -> Shared<Barrier> {
        intrinsic!(|scope| {
            let variable =
                scope.create_shared(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Cube), None);
            scope.register(BarrierOps::Init {
                barrier: variable,
                is_elected: is_elected.expand,
                arrival_count: arrival_count.expand,
            });
            variable.into()
        })
    }

    /// Create a shared memory barrier that can be accesses by all units in the cube. Only declared,
    /// but not initialized.
    pub fn shared_uninit() -> Shared<Barrier> {
        intrinsic!(|scope| {
            let variable =
                scope.create_shared(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Cube), None);
            scope.register(BarrierOps::Declare { barrier: variable });
            variable.into()
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
    #[allow(unused_variables)]
    pub fn init_manual(&self, arrival_count: u32) {
        intrinsic!(|scope| {
            let barrier = self.expand;

            scope.register(BarrierOps::InitManual {
                barrier,
                arrival_count: arrival_count.expand,
            });
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
    #[allow(unused_variables)]
    pub fn memcpy_async<C: CubePrimitive>(&self, source: &[C], destination: &mut [C]) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let source_length = source.__extract_length(scope).expand;
            let source = unsafe { *source.__expand_as_ptr_method(scope) }.expand;
            let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.expand;

            let mem_copy = BarrierOps::MemCopyAsync {
                barrier,
                destination,
                source,
                source_length,
            };

            scope.register(Instruction::no_out(mem_copy));
        })
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    #[allow(unused_variables)]
    pub fn memcpy_async_cooperative<C: CubePrimitive>(&self, source: &[C], destination: &mut [C]) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let source_length = source.__extract_length(scope).expand;
            let source = unsafe { *source.__expand_as_ptr_method(scope) }.expand;
            let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.expand;

            let mem_copy = BarrierOps::MemCopyAsyncCooperative {
                barrier,
                source,
                destination,
                source_length,
            };

            scope.register(Instruction::no_out(mem_copy));
        })
    }

    /// Copy the source slice to destination. Uses transaction count like TMA, so use with
    /// `expect_tx` or `arrive_and_expect_tx`.
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    #[allow(unused_variables)]
    pub fn memcpy_async_tx<C: CubePrimitive>(&self, source: &[C], destination: &mut [C]) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let source_length = source.__extract_length(scope).expand;
            let source = unsafe { *source.__expand_as_ptr_method(scope) }.expand;
            let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.expand;

            let mem_copy = BarrierOps::MemCopyAsyncTx {
                barrier,
                source,
                destination,
                source_length,
            };

            scope.register(Instruction::no_out(mem_copy));
        })
    }
}

// Arrival and Wait

#[cube]
impl Barrier {
    /// Arrive at the barrier, decrementing arrival count
    pub fn arrive(&self) -> BarrierToken {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type() else {
                unreachable!()
            };
            let token = scope.create_barrier_token(barrier.index().unwrap(), level);
            scope.register(Instruction::new(BarrierOps::Arrive { barrier }, token));
            token.into()
        })
    }

    /// Arrive at the barrier, decrementing arrival count. Additionally increments expected count.
    #[allow(unused_variables)]
    pub fn arrive_and_expect_tx(&self, arrival_count: u32, transaction_count: u32) -> BarrierToken {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type() else {
                unreachable!()
            };
            let token = scope.create_barrier_token(barrier.index().unwrap(), level);
            let arrival_count: Variable = arrival_count.into();
            let transaction_count: Variable = transaction_count.into();
            scope.register(Instruction::new(
                BarrierOps::ArriveTx {
                    barrier,
                    arrive_count_update: arrival_count,
                    transaction_count_update: transaction_count,
                },
                token,
            ));
            token.into()
        })
    }

    /// Increments the expected count of the barrier.
    #[allow(unused_variables)]
    pub fn expect_tx(&self, expected_count: u32) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let transaction_count: Variable = expected_count.into();
            scope.register(BarrierOps::ExpectTx {
                barrier,
                transaction_count_update: transaction_count,
            });
        })
    }

    /// Wait until all data is loaded
    pub fn arrive_and_wait(&self) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            scope.register(BarrierOps::ArriveAndWait { barrier });
        })
    }

    /// Wait at the barrier until all arrivals are done
    #[allow(unused_variables)]
    pub fn wait(&self, token: BarrierToken) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let token = token.expand;
            scope.register(BarrierOps::Wait { barrier, token });
        })
    }

    /// Wait at the barrier until the `phase` is completed. Doesn't require a token, but needs phase
    /// to be managed manually.
    #[allow(unused_variables)]
    pub fn wait_parity(&self, phase: u32) {
        intrinsic!(|scope| {
            let barrier = self.expand;
            let phase = phase.expand;
            scope.register(BarrierOps::WaitParity { barrier, phase });
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
        let source_length = copy_length.into();
        let source = unsafe { *source.__expand_as_ptr_method(scope) }.expand;
        let destination = unsafe { *destination.__expand_as_ptr_method(scope) }.expand;
        let scalar_size = C::__expand_as_type(scope).storage_type().size();

        let mem_copy = BarrierOps::CopyAsync {
            source,
            destination,
            source_length,
            copy_length: copy_length * scalar_size as u32,
            checked: false,
        };

        scope.register(Instruction::no_out(mem_copy));
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
        let source_length = source.__extract_length(scope).expand;

        // OOB pointer is allowed as long as length is 0
        let source = unsafe { (*source.__expand_as_ptr_unchecked_method(scope)).expand };
        let destination = unsafe { (*destination.__expand_as_ptr_unchecked_method(scope)).expand };
        let scalar_size = C::__expand_as_type(scope).storage_type().size();

        let mem_copy = BarrierOps::CopyAsync {
            source,
            destination,
            source_length,
            copy_length: copy_length * scalar_size as u32,
            checked: true,
        };

        scope.register(Instruction::no_out(mem_copy));
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
            let barrier = self.expand;
            let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type() else {
                unreachable!()
            };
            let token = scope.create_barrier_token(barrier.index().unwrap(), level);
            scope.register(Instruction::new(
                BarrierOps::CommitCopyAsync { barrier },
                token,
            ));
        })
    }
}

impl From<SharedExpand<Barrier>> for BarrierExpand {
    fn from(value: SharedExpand<Barrier>) -> Self {
        value.expand.into()
    }
}
