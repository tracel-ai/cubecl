use cubecl_macros_internal::cube_op;
use derive_more::From;
use derive_new::new;
use pliron::derive::{format, op_interface_impl, pliron_attr};

use crate::{interfaces::Synchronizes, prelude::*};

#[format]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum SyncScope {
    #[format("`plane`")]
    Plane,
    #[format("`cube`")]
    Cube,
    #[format("`device`")]
    Device,
}

#[pliron_attr(name = "cube.sync_scope", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord)]
pub struct SyncScopeAttr(pub SyncScope);

#[cube_op(name = "sync.sync")]
#[result_ty(none)]
pub struct SyncOp {
    pub scope: SyncScopeAttr,
}

#[op_interface_impl]
impl Synchronizes for SyncOp {
    fn scope(&self, ctx: &Context) -> SyncScope {
        self.get_attr_scope(ctx).unwrap().0
    }
}

/// Fences the async proxy in CUDA, to make shared memory available to it. Does not implement
/// `Synchronizes`, because it works only as a memory availability barrier with an outside chip.
/// It does not synchronize the actual threads, and is typically called only by the TMA leader.
#[cube_op(name = "sync.sync_async_proxy")]
#[result_ty(none)]
pub struct SyncAsyncProxyOp {}
