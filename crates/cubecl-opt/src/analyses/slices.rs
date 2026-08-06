use cubecl_ir::prelude::*;
use pliron::{
    linked_list::ContainsLinkedList,
    utils::table::{HSet, ISet},
};

fn get_forward_slice_impl(
    ctx: &Context,
    op: Ptr<Operation>,
    visited: &mut HSet<Ptr<Operation>>,
    forward_slice: &mut ISet<Ptr<Operation>>,
) {
    for region in op.regions(ctx) {
        for block in region.deref(ctx).iter(ctx) {
            for block_op in block.deref(ctx).iter(ctx) {
                if !forward_slice.contains(&block_op) {
                    visited.insert(block_op);
                    get_forward_slice_impl(ctx, block_op, visited, forward_slice);
                    visited.remove(&block_op);
                }
            }
        }
    }

    for result in op.results(ctx) {
        for r#use in result.uses(ctx) {
            let user_op = r#use.user_op();
            if !forward_slice.contains(&user_op) && visited.insert(user_op) {
                get_forward_slice_impl(ctx, user_op, visited, forward_slice);
                visited.remove(&user_op);
            }
        }
    }

    forward_slice.insert(op);
}

pub fn get_value_forward_slice(ctx: &Context, root: Value) -> ISet<Ptr<Operation>> {
    let mut forward_slice = ISet::default();
    let mut visited = HSet::default();
    for r#use in root.uses(ctx) {
        let user = r#use.user_op();
        visited.insert(user);
        get_forward_slice_impl(ctx, user, &mut visited, &mut forward_slice);
        visited.remove(&user);
    }

    forward_slice.reverse();
    forward_slice
}
