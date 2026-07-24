use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::{
    derive::{format, op_interface_impl, pliron_attr},
    opts::dce::SideEffects,
};

use crate::{CanMaterialize, NoMemoryEffect, interfaces::Synchronizes, prelude::*};

/// Scope that the synchronization should apply to. This is a *minimum*, when fine-grained control
/// is not available it should synchronize at the smallest scope that includes this scope
/// (i.e. `SyncScope::Plane` may be implemented by a `workgroupBarrier()`)
#[format]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum SyncScope {
    Plane,
    Cube,
    Device,
}

#[pliron_attr(name = "cube.sync_scope", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord)]
pub struct SyncScopeAttr(pub SyncScope);

#[cube_op(name = "sync.sync")]
#[result_ty(none)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct SyncOp {
    pub scope: SyncScopeAttr,
}

#[op_interface_impl]
impl Synchronizes for SyncOp {
    fn scope(&self, ctx: &Context) -> SyncScope {
        self.scope(ctx).0
    }
}

#[op_interface_impl]
impl SideEffects for SyncOp {
    fn has_side_effects(&self, _ctx: &Context) -> bool {
        true
    }
}

/// Fences the async proxy in CUDA, to make shared memory available to it. Does not implement
/// `Synchronizes`, because it works only as a memory availability barrier with an outside chip.
/// It does not synchronize the actual threads, and is typically called only by the TMA leader.
#[cube_op(name = "sync.sync_async_proxy")]
#[result_ty(none)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct SyncAsyncProxyOp {}
