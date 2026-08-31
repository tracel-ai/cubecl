use super::prelude::*;
use pliron::builtin::op_interfaces::OneResultInterface;

/// Insert `op` at the rewriter's point and return the value it defines.
///
/// Building an op and immediately wanting its result is the shape of nearly every line
/// in a lowering, and spelling it out three times per line is what makes those files
/// hard to read past.
pub fn insert<O: Op + OneResultInterface>(
    ctx: &mut Context,
    inserter: &mut impl Inserter,
    op: &O,
) -> Value {
    inserter.insert_op(ctx, op);
    op.get_result(ctx)
}
