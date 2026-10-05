//! Aliasing and access attributes on a GPU kernel's buffer parameters.

use crate::{
    prelude::BufferIOAttr,
    shared::llvm_module::{EntryFunction, Param},
};

/// Marks the buffer parameters `noalias`, and the read-only ones `readonly`, from the recorded
/// access modes, wherever no cube writes them within the launch; the metadata gets both.
/// Distinct bindings never overlap, which is what lets the backend keep a value in a register
/// across a store, and a read-only pointer is what lets NVPTX load through the non-coherent cache
/// and AMDGPU prove a uniform load is not clobbered and make it a scalar load.
///
/// A kernel that loads an atomic reads what another cube wrote within the launch, a turn handed
/// on through a counter ([`AtomicReads`]). A buffer it writes may then be written by another cube
/// between its own accesses, and one it loads atomically by someone, so neither is promised
/// anything: with the attributes, AMDGPU makes a read after the acquire a scalar load, served
/// from a cache the acquire does not invalidate. The buffers it only reads, which no cube writes,
/// keep both.
///
/// `entry`'s parameters are the buffers in binding order followed by `metadata_params`
/// metadata parameters. An LLVM without one of the two attributes gets the other alone.
pub(crate) fn annotate_buffer_params(
    entry: &EntryFunction<'_>,
    io: &[BufferIOAttr],
    metadata_params: u32,
) {
    let atomic_reads = AtomicReads::of(entry);

    let (buffers, metadata) = entry.split_params(metadata_params);
    let buffers = buffers.enumerate().map(|(binding, param)| {
        let promise = Promise::of_buffer(io.get(binding).copied(), param, &atomic_reads);
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

/// The buffers a kernel's atomic loads read.
#[derive(Debug, PartialEq)]
enum AtomicReads {
    /// It loads no atomic: no cube reads what another wrote within the launch.
    None,
    /// Each atomic load's pointer traced back to its parameter.
    Traced(Vec<Param>),
    /// Some atomic load's pointer passes through what the trace does not follow, so any
    /// buffer may be one of them.
    Untraced,
}

impl AtomicReads {
    fn of(entry: &EntryFunction<'_>) -> Self {
        let mut traced = Vec::new();
        for pointer in entry
            .instructions()
            .filter_map(|inst| inst.atomically_loaded())
        {
            match entry.param_under(pointer) {
                Some(param) => traced.push(param),
                None => return Self::Untraced,
            }
        }
        match traced.is_empty() {
            true => Self::None,
            false => Self::Traced(traced),
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
    /// The promise for a buffer of access mode `io` at `param`, in a kernel whose atomic loads
    /// read `atomic_reads`. A buffer with no recorded mode is taken as read and written.
    fn of_buffer(io: Option<BufferIOAttr>, param: Param, atomic_reads: &AtomicReads) -> Self {
        let io = io.unwrap_or(BufferIOAttr::ReadWrite);
        let read_only = io == BufferIOAttr::ReadOnly;
        match atomic_reads {
            AtomicReads::None if read_only => Self::DistinctReadOnly,
            AtomicReads::None => Self::Distinct,
            AtomicReads::Traced(loaded) if loaded.contains(&param) => Self::Nothing,
            AtomicReads::Traced(_) | AtomicReads::Untraced if io.is_writable() => Self::Nothing,
            AtomicReads::Traced(_) if read_only => Self::DistinctReadOnly,
            // An untraced load may read it, so it keeps only what the load leaves true.
            AtomicReads::Traced(_) | AtomicReads::Untraced => Self::Distinct,
        }
    }
}
