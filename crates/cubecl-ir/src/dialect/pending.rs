//! Commit groups of asynchronous operations, waited on through tokens.
//!
//! The hardware counts committed groups: a wait names how many of the newest groups may still
//! run, not which group it waits for. The kernel instead commits a group into a token and waits
//! on the token, and `ResolvePendingPass` turns each wait into the count of groups committed
//! after the token's group, on the path with the fewest.

use cubecl_macros_internal::cube_op;

use crate::{
    CanMaterialize, HasSideEffects, Pure,
    prelude::*,
    types::pending::{AsyncGroup, GroupTokenType},
};

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

/// Waits until the group `token` was committed into completes.
#[cube_op(name = "pending.wait")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct WaitOp {
    pub token: Value,
}
