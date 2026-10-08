//! Commit groups of asynchronous operations, waited on through tokens.
//!
//! The hardware counts committed groups: a wait names how many of the newest groups may still
//! run, not which group it waits for. The kernel instead commits a group into a token and waits
//! on the token, and `ResolvePendingPass` turns each wait into the count of groups committed
//! after the token's group, on the path with the fewest.

use cubecl_macros_internal::cube_op;

use crate::{
    CanMaterialize, HasSideEffects, Pure,
    dialect::{
        matrix::{WgmmaCommitGroupOp, WgmmaWaitGroupOp},
        tma::{CommitGroupOp, WaitGroupReadOp},
    },
    prelude::*,
    types::pending::{AsyncGroup, GroupTokenType},
};

impl AsyncGroup {
    /// The operation that commits a group of this kind.
    pub fn commit_op(self, ctx: &mut Context) -> Ptr<Operation> {
        match self {
            AsyncGroup::Warpgroup => WgmmaCommitGroupOp::new(ctx).get_operation(),
            AsyncGroup::BulkCopy => CommitGroupOp::new(ctx).get_operation(),
        }
    }

    /// The operation that waits until at most `max_pending` groups of this kind are running.
    pub fn wait_op(self, ctx: &mut Context, max_pending: usize) -> Ptr<Operation> {
        match self {
            AsyncGroup::Warpgroup => WgmmaWaitGroupOp::new(ctx, max_pending).get_operation(),
            AsyncGroup::BulkCopy => WaitGroupReadOp::new(ctx, max_pending).get_operation(),
        }
    }
}

fn token_ty(ctx: &Context, group: &AsyncGroup) -> TypeHandle {
    GroupTokenType::get(ctx, *group).into()
}

/// Commits every operation of `group` issued and not yet committed into a new group, and
/// returns its token.
#[cube_op(name = "pending.commit")]
#[result_ty(from_inputs = |ctx, group: &AsyncGroup| token_ty(ctx, group))]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct CommitOp {
    pub group: AsyncGroup,
}

/// A token of `group` with nothing to wait for. Waiting on it is free.
#[cube_op(name = "pending.ready")]
#[result_ty(from_inputs = |ctx, group: &AsyncGroup| token_ty(ctx, group))]
#[op_traits(CanMaterialize, Pure)]
pub struct ReadyOp {
    pub group: AsyncGroup,
}

/// Waits until the group `token` was committed into completes, as its [`AsyncGroup`] defines.
#[cube_op(name = "pending.wait")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct WaitOp {
    pub token: Value,
}
