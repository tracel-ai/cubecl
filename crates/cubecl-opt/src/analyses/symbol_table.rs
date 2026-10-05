use cubecl_ir::interfaces::control_flow::SymbolOpInterface;
use cubecl_ir::prelude::*;
use pliron::linked_list::ContainsLinkedList;

pub fn walk_symbol_tables<State>(
    ctx: &Context,
    state: &mut State,
    op: Ptr<Operation>,
    mut all_sym_uses_visible: bool,
    callback: for<'a> fn(&Context, &mut State, &'a dyn SymbolTableInterface, bool),
) {
    if op.impls::<dyn SymbolTableInterface>(ctx)
        && let Some(symbol) = op_cast::<dyn SymbolOpInterface>(&*op.dyn_op(ctx))
    {
        all_sym_uses_visible |= symbol.is_private(ctx);
    } else {
        all_sym_uses_visible = true;
    }

    for region in op.regions(ctx) {
        for block in region.deref(ctx).iter(ctx) {
            for nested_op in block.deref(ctx).iter(ctx) {
                walk_symbol_tables(ctx, state, nested_op, all_sym_uses_visible, callback);
            }
        }
    }

    if let Some(symbol_table) = op_cast::<dyn SymbolTableInterface>(&*op.dyn_op(ctx)) {
        callback(ctx, state, symbol_table, all_sym_uses_visible);
    }
}
