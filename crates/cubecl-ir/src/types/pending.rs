//! The token a committed group of asynchronous operations is waited on through.

use pliron::derive::{pliron_attr, pliron_type};

use crate::{aligned, sized};

/// A kind of asynchronous operation that completes in commit groups: each commit closes a group
/// of the operations issued since the last one, and a wait lets at most a number of the newest
/// groups of the same kind still run.
#[pliron_attr(name = "cube.async_group", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AsyncGroup {
    /// Warpgroup MMAs. A group completes once its MMAs wrote their accumulators.
    Warpgroup,
    /// Bulk copies from shared memory, TMA stores among them. A group completes once its copies
    /// read their shared memory, which may then be written again. Their writes to global memory
    /// may still be in flight, and are visible once the kernel ends.
    BulkCopy,
}

/// A committed group of asynchronous operations, waited on until it completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cube.group_token",
    format = "`group_token<` $0 `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct GroupTokenType(pub AsyncGroup);
// Never stored: the token is resolved away before any target lowering. A size lets a variable
// hold it until then.
aligned!(GroupTokenType, 4);
sized!(GroupTokenType, 4);
