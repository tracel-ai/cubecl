//! Software graph capture for wgpu.
//!
//! WebGPU has no driver-side graph object and no re-submittable command
//! buffers (`queue.submit` consumes them), so unlike CUDA/HIP a captured graph
//! is a **software graph**: everything a launch resolves per dispatch —
//! pipeline lookup, binding resolution, info-uniform upload, bind group
//! creation — is done once while recording, and
//! [`WgpuStream::replay_graph`](super::stream::WgpuStream::replay_graph)
//! re-encodes the prebuilt state in a tight loop.
//!
//! Replay therefore stays O(n) in recorded tasks — encoding cannot be skipped
//! under WebGPU — and only the constant shrinks. That makes it worth measuring
//! rather than assuming: `cargo run --release -p cubecl-wgpu --example
//! graph_bench` sweeps kernel size against graph length and prints, per size,
//! how much of the pass was launch overhead in the first place. The win is
//! confined to sizes where that share is large, and the sweep includes
//! GPU-bound rows where it must vanish.

use crate::WgpuResource;
use crate::schedule::Addresses;
use cubecl_runtime::memory_management::{
    ManagedMemoryHandle, ManagedMemoryId, SharedMemoryBindings,
};
use std::sync::Arc;
use wgpu::ComputePipeline;

/// A captured graph: the recorded launch sequence, fully resolved (see the
/// [module docs](self)).
///
/// Owned by the [`WgpuServer`](super::server::WgpuServer) registry and
/// referenced by [`GraphId`](cubecl_runtime::id::GraphId); the client
/// references the graph by id and, on the last drop, asks the server to
/// release it.
#[derive(Debug)]
pub struct WgpuGraph {
    /// The recorded tasks, replayed in order.
    pub(crate) tasks: Vec<ReplayTask>,
    /// Every pool slice the capture window allocated (intermediates, info
    /// uniforms, Vulkan address buffers), pinned so the pools cannot reuse
    /// memory a replay still runs against. Dropped with the graph.
    pub(crate) _retained: Vec<ManagedMemoryHandle>,
    /// Cross-stream input bindings the recorded tasks reference, pinned for
    /// the graph's lifetime instead of until the next submission, which is
    /// where [`WgpuStream::flush`](super::stream::WgpuStream::flush) releases
    /// them on the normal path.
    pub(crate) _shared: SharedMemoryBindings,
    /// The buffers the recorded launches were given, deduplicated. A replay
    /// that fails runs none of those launches, so it leaves every one of these
    /// as it was — which is what a later read of one has to fail on, whichever
    /// stream asks (see
    /// [`StreamErrors::push_unwritten`](cubecl_runtime::stream::StreamErrors::push_unwritten)).
    pub(crate) unwritten: Vec<ManagedMemoryId>,
}

/// One recorded dispatch, resolved down to what `wgpu` needs at encode time.
#[derive(Debug)]
pub(crate) struct ReplayTask {
    pub(crate) pipeline: Arc<ComputePipeline>,
    /// Built once at record time; bind groups are reusable and the buffers
    /// they reference are pinned by the graph. `None` when the kernel binds
    /// no resources (Vulkan immediate-address mode).
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    /// Vulkan buffer device addresses passed as immediates, resolved once at
    /// record time.
    ///
    /// Addresses into slices the capture window allocated stay valid because
    /// `_retained` pins them. An address into a buffer the *caller* owns is the
    /// caller's to keep alive, which is what
    /// [`Graph::replay`](cubecl_runtime::client::Graph::replay)'s liveness
    /// requirement is about — the same contract that lets a replay pick up
    /// bytes rewritten between replays.
    pub(crate) immediates: Option<Addresses>,
    /// Buffers needing an explicit transition to storage read-write state
    /// (Vulkan buffer-address mode, where usage tracking cannot see them).
    pub(crate) transitions: Vec<WgpuResource>,
    pub(crate) dispatch: ReplayDispatch,
}

/// The dispatch shape of a recorded task.
#[derive(Debug)]
pub(crate) enum ReplayDispatch {
    Static(u32, u32, u32),
    /// Indirect dispatch: the workgroup count is read from this buffer at
    /// execution time rather than baked in, so replays pick up counts written
    /// between them. Pinned by `_retained` when the capture window allocated
    /// it; otherwise kept alive by the caller, as for
    /// [`ReplayTask::immediates`].
    Dynamic(WgpuResource),
}

/// The in-progress recording on a stream, moved into a [`WgpuGraph`] at
/// `end_capture`.
#[derive(Debug, Default)]
pub(crate) struct GraphRecording {
    pub(crate) tasks: Vec<ReplayTask>,
    /// Cross-stream input bindings of recorded tasks (see [`WgpuGraph::_shared`]).
    pub(crate) shared: SharedMemoryBindings,
    /// Uniform slices created inside the window, held alive until
    /// `end_capture` so the memory manager's `capture_end` retains them on
    /// the graph — retention only covers slices still live at that point.
    pub(crate) uniform_pins: Vec<ManagedMemoryHandle>,
    /// The buffers the recorded launches were given, in launch order and with
    /// repeats (see [`WgpuGraph::unwritten`], which is this deduplicated).
    pub(crate) buffers: Vec<ManagedMemoryId>,
}
