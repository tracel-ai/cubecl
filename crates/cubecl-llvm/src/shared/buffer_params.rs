//! Aliasing and access attributes on a GPU kernel's buffer parameters.

use crate::{
    prelude::{AddressSpace, BufferIOAttr, Context, CubePointerType, Op, Operation, Ptr},
    shared::llvm_module::EntryFunction,
};
use cubecl_core::ir::{
    dialect::{atomic::AtomicLoadOp, base::OperationPtrExt},
    interfaces::side_effects::{MemoryEffect, MemoryEffectsOp},
};
use pliron::{
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::op_cast,
    r#type::Typed,
};

/// Marks the buffer parameters `noalias`, and the read-only ones `readonly`, from the recorded
/// access modes, wherever no cube writes them within the launch; the metadata gets both.
/// Distinct bindings never overlap, which is what lets the backend keep a value in a register
/// across a store, and a read-only pointer is what lets NVPTX load through the non-coherent cache
/// and AMDGPU prove a uniform load is not clobbered and make it a scalar load.
///
/// A kernel whose atomics read global memory reads what another cube wrote within the launch, a
/// turn handed on through a counter ([`AtomicReads`]). A buffer it writes may then be written by
/// another cube between its own accesses, and one an atomic reads by someone, so neither is
/// promised anything: with the attributes, AMDGPU makes a read after the acquire a scalar load,
/// served from a cache the acquire does not invalidate. The buffers it only reads, which no cube
/// writes, keep both.
///
/// `entry`'s parameters are the buffers in binding order followed by `metadata_params`
/// metadata parameters. An LLVM without one of the two attributes gets the other alone.
pub(crate) fn annotate_buffer_params(
    entry: &EntryFunction<'_>,
    io: &[BufferIOAttr],
    atomic_reads: &AtomicReads,
    metadata_params: u32,
) {
    let (buffers, metadata) = entry.split_params(metadata_params);
    let buffers = buffers.enumerate().map(|(binding, param)| {
        let promise = Promise::of_buffer(io.get(binding).copied(), binding, atomic_reads);
        (param, promise)
    });
    let metadata = metadata.map(|param| (param, Promise::DistinctReadOnly));

    for (param, promise) in buffers.chain(metadata) {
        if !entry.param_is_pointer(param) {
            continue;
        }
        match promise {
            Promise::Nothing => {}
            Promise::Distinct => {
                let _ = entry.add_param_attribute(param, "noalias", 0);
            }
            Promise::DistinctReadOnly => {
                let _ = entry.add_param_attribute(param, "noalias", 0);
                let _ = entry.add_param_attribute(param, "readonly", 0);
            }
        }
    }
}

/// The global buffers a kernel's atomics read: a load, a read-modify-write or a compare-exchange,
/// each of which observes what another cube stored. An atomic store alone observes nothing, and
/// a shared or local atomic only what its own cube stored.
#[derive(Debug, Default, Clone, PartialEq)]
pub enum AtomicReads {
    /// No atomic reads global memory: no cube reads what another wrote within the launch.
    #[default]
    None,
    /// The bindings the atomics read, each named by its pointer's address space.
    Bindings(Vec<usize>),
    /// Some atomic reads through a value that is not a pointer, so any buffer may be one.
    Unattributed,
}

impl AtomicReads {
    /// Read off the cube dialect under `root`, before it is lowered, while every pointer's type
    /// still names the binding it addresses.
    pub(crate) fn of(ctx: &Context, root: Ptr<Operation>) -> Self {
        let mut reads = Self::None;
        walk_op(
            ctx,
            &mut reads,
            &WALKCONFIG_PREORDER_FORWARD,
            root,
            |ctx, reads, node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                let op = op.dyn_op(ctx);
                if op.get_opid().dialect != AtomicLoadOp::get_opid_static().dialect {
                    return;
                }
                let Some(effects) = op_cast::<dyn MemoryEffectsOp>(op.as_ref()) else {
                    return;
                };
                for effect in effects.memory_effects(ctx) {
                    if let MemoryEffect::Read(ptr) = effect {
                        reads.add(ptr.get_type(ctx).deref(ctx).downcast_ref());
                    }
                }
            },
        );
        reads
    }

    fn add(&mut self, pointer: Option<&CubePointerType>) {
        let binding = match pointer {
            Some(CubePointerType {
                address_space: AddressSpace::Global(binding),
                ..
            }) => *binding,
            Some(_) => return,
            None => {
                *self = Self::Unattributed;
                return;
            }
        };
        match self {
            Self::None => *self = Self::Bindings(vec![binding]),
            Self::Bindings(bindings) if !bindings.contains(&binding) => bindings.push(binding),
            Self::Bindings(_) | Self::Unattributed => {}
        }
    }
}

/// What one buffer parameter is promised.
#[derive(Debug, PartialEq)]
enum Promise {
    /// Neither attribute: some cube may write it within the launch.
    Nothing,
    /// `noalias`.
    Distinct,
    /// `noalias` and `readonly`.
    DistinctReadOnly,
}

impl Promise {
    /// The promise for the buffer of access mode `io` at `binding`, in a kernel whose atomics
    /// read `atomic_reads`. A buffer with no recorded mode is taken as read and written.
    fn of_buffer(io: Option<BufferIOAttr>, binding: usize, atomic_reads: &AtomicReads) -> Self {
        let io = io.unwrap_or(BufferIOAttr::ReadWrite);
        let read_only = io == BufferIOAttr::ReadOnly;
        match atomic_reads {
            AtomicReads::None if read_only => Self::DistinctReadOnly,
            AtomicReads::None => Self::Distinct,
            AtomicReads::Unattributed => Self::Nothing,
            AtomicReads::Bindings(read) if read.contains(&binding) => Self::Nothing,
            AtomicReads::Bindings(_) if io.is_writable() => Self::Nothing,
            AtomicReads::Bindings(_) if read_only => Self::DistinctReadOnly,
            AtomicReads::Bindings(_) => Self::Distinct,
        }
    }
}
