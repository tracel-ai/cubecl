use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::derive::{format, op_interface_impl, pliron_attr};

use crate::{CanMaterialize, HasSideEffects, interfaces::Synchronizes, prelude::*};

/// Scope that the synchronization should apply to. This is a *minimum*, when fine-grained control
/// is not available it should synchronize at the smallest scope that includes this scope
/// (i.e. `SyncScope::Plane` may be implemented by a `workgroupBarrier()`)
///
/// Every scope up to [`Cube`](SyncScope::Cube) synchronizes the units that share it and orders the
/// memory they wrote for each other. [`Device`](SyncScope::Device) contains
/// [`Cube`](SyncScope::Cube) — the units of the cube meet at it and their shared memory is
/// ordered — and reaches past it: it is also a release and an acquire at device scope, so a write
/// one cube published before it is visible to any other cube that synchronizes at this scope
/// afterwards. Only a runtime whose
/// [`device_memory_scope`](crate::Features::device_memory_scope) is set promises that second
/// half; the others give the cube barrier alone.
#[format]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum SyncScope {
    Unit,
    Plane,
    Cube,
    Device,
}

#[pliron_attr(name = "cube.sync_scope", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord)]
pub struct SyncScopeAttr(pub SyncScope);

#[cube_op(name = "sync.sync")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct SyncOp {
    pub scope: SyncScopeAttr,
}

#[op_interface_impl]
impl Synchronizes for SyncOp {
    fn minimum_scope(&self, ctx: &Context) -> SyncScope {
        self.scope(ctx).0
    }
    fn maximum_scope(&self, ctx: &Context) -> SyncScope {
        self.scope(ctx).0
    }
}

/// Fences the async proxy in CUDA, to make shared memory available to it. Does not implement
/// `Synchronizes`, because it works only as a memory availability barrier with an outside chip.
/// It does not synchronize the actual threads, and is typically called only by the TMA leader.
#[cube_op(name = "sync.sync_async_proxy")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct SyncAsyncProxyOp {}

/// Waits until every kernel this one depends on has completed and its memory is visible. Only
/// does something when the kernel was launched with programmatic dependent launch, which lets it
/// start before those kernels finish; otherwise they have already finished by the time it runs.
/// PTX: `griddepcontrol.wait`
#[cube_op(name = "sync.wait_for_prerequisite_kernels")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct WaitForPrerequisiteKernelsOp {}

/// Lets the kernels launched after this one with programmatic dependent launch start once every
/// cube of this kernel has called it or exited. A scheduling hint that publishes no memory: the
/// dependent kernel's [`WaitForPrerequisiteKernelsOp`] is what waits for this kernel's writes.
/// PTX: `griddepcontrol.launch_dependents`
#[cube_op(name = "sync.allow_dependent_kernels_to_launch")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct AllowDependentKernelsToLaunchOp {}
