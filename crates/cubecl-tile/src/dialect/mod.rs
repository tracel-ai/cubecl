//! The tile ops in the IR, each lowered by every backend's `LowerOpsPass`.

mod drain;
mod level;
#[cfg(feature = "sequential-levels")]
mod sequential;

pub use drain::DrainOp;
pub use level::LevelOp;

use cubecl_ir::prelude::*;
use pliron::linked_list::ContainsLinkedList;

/// Panics where a value `op`'s region defines is used outside it: an op that yields nothing
/// stands for writes alone, so what its region computes stays inside it, and a backend may lower
/// the op without keeping any of it.
pub(crate) fn assert_nothing_escapes(ctx: &Context, op: Ptr<Operation>, name: &str) {
    let inside = |user: Ptr<Operation>| {
        let mut at = Some(user);
        while let Some(current) = at {
            if current == op {
                return true;
            }
            at = current.deref(ctx).get_parent_op(ctx);
        }
        false
    };
    let mut pending = vec![op];
    while let Some(outer) = pending.pop() {
        let outer = outer.deref(ctx);
        for region in outer.regions() {
            for block in region.deref(ctx).iter(ctx) {
                for inner in block.deref(ctx).iter(ctx) {
                    for result in inner.deref(ctx).results() {
                        assert!(
                            result.uses(ctx).iter().all(|used| inside(used.user_op())),
                            "{name}: a value its region computes is used after it; the op yields \
                             nothing, so what it computes must stay inside it"
                        );
                    }
                    pending.push(inner);
                }
            }
        }
    }
}
