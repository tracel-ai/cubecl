//! Work a kernel issued that completes later.
//!
//! Some operations run asynchronously: a warpgroup MMA, a TMA store. The kernel goes on while
//! they run, and must not touch what they read or write until they complete. Such an operation
//! returns a [`Pending`], which holds what the work produces, or `()`, and gives it back from
//! [`Pending::wait`] once the work is done with what the kernel must not touch:
//!
//! ```rust, ignore
//! let stored = tma_store_2d(&tile, &mut output, row, col);
//! // ... other work, while the store reads `tile` ...
//! stored.wait();
//! // `tile` may be written again.
//! ```
//!
//! The hardware tracks such work in commit groups, and waits for a number of groups rather than
//! for a given one. The compiler derives that number for each wait, from the groups committed
//! between the operation and the wait on every path that leads to it: a wait in a pipelined loop
//! lets the newer groups run on.
//!
//! What "done" means depends on the work. A warpgroup MMA is done once it wrote its accumulator.
//! A TMA store is done once it read its shared memory source: its writes to global memory may
//! still be in flight, and are visible once the kernel ends. [`Pending::wait_complete`] also
//! waits for those writes, for a kernel that hands them to another cube.

use crate::{self as cubecl, prelude::*};
use cubecl_ir::{
    dialect::pending::{CommitOp, ReadyOp, WaitOp},
    types::pending::{AsyncGroup, WaitUntil},
};
use cubecl_macros::{CubeTypeMut, intrinsic};
use pliron::value::Value;

/// Work that completes later, and the `T` it produces then.
#[derive(CubeType, CubeTypeMut)]
#[must_use = "the work runs on until it is waited on"]
pub struct Pending<T: PendingValue> {
    #[allow(unused)]
    pub(crate) value: T,
    pub(crate) token: GroupToken,
}

/// A value [`Pending`] work produces.
#[cube]
pub trait PendingValue: CubeType + Sized {
    /// Waits for `pending` to complete, and returns its value.
    fn wait_for(pending: Pending<Self>) -> Self;
}

#[cube]
impl<T: PendingValue> Pending<T> {
    /// Waits for the work to complete, and returns what it produced.
    pub fn wait(self) -> T {
        T::wait_for(self)
    }
}

#[cube]
impl PendingValue for () {
    fn wait_for(pending: Pending<Self>) -> Self {
        pending.token.wait();
    }
}

#[cube]
impl Pending<()> {
    /// Waits for the work to fully complete. A TMA store has then also performed its writes to
    /// global memory, which a fence or an atomic after the wait publishes to other cubes.
    /// Anything else completes as [`Pending::wait`] waits for it.
    pub fn wait_complete(self) {
        self.token.wait_until(comptime![WaitUntil::Complete]);
    }
}

// A completion carries no value, so waiting on it leaves nothing behind: a loop may wait on it
// and replace it in the same iteration.
impl Copy for PendingExpand<()> {}
impl Clone for PendingExpand<()> {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Pending<()> {}
impl Clone for Pending<()> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PendingValue> PendingExpand<T> {
    /// Commits the operations of `group` issued since its last commit, which produce `value`.
    pub(crate) fn commit(scope: &Scope, group: AsyncGroup, value: T::ExpandType) -> Self {
        Self {
            value,
            token: GroupToken::commit(scope, group),
        }
    }

    /// `value`, with nothing of `group` to wait for.
    pub(crate) fn ready(scope: &Scope, group: AsyncGroup, value: T::ExpandType) -> Self {
        let op = ReadyOp::new(scope.ctx_mut(), group);
        Self {
            value,
            token: scope.register_with_result(&op).into(),
        }
    }
}

/// The token of a commit group, which a wait resolves to a count of groups.
#[derive(Clone, Copy)]
pub(crate) struct GroupToken;
pub(crate) type GroupTokenExpand = NativeExpand<GroupToken>;

impl GroupToken {
    pub(crate) fn commit(scope: &Scope, group: AsyncGroup) -> NativeExpand<GroupToken> {
        let op = CommitOp::new(scope.ctx_mut(), group);
        scope.register_with_result(&op).into()
    }
}

#[cube]
impl GroupToken {
    /// Waits until the group this token was committed into is done with what the kernel must
    /// not touch.
    pub(crate) fn wait(&self) {
        self.wait_until(comptime![WaitUntil::Released]);
    }

    /// Waits until the group this token was committed into reaches `until`.
    #[allow(unused_variables)]
    pub(crate) fn wait_until(&self, #[comptime] until: WaitUntil) {
        intrinsic!(|scope| {
            let token = self.read_value(scope);
            scope.register(&WaitOp::new(scope.ctx_mut(), token, until));
        })
    }
}

impl CubeType for GroupToken {
    type ExpandType = NativeExpand<GroupToken>;
}

impl CubeDebug for GroupToken {}

impl ReadValue for NativeExpand<GroupToken> {
    fn read_value(&self, scope: &Scope) -> Value {
        self.expand.read_value(scope)
    }
}

impl NativeAssign for GroupToken {}

impl AsMutExpand for NativeExpand<GroupToken> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
