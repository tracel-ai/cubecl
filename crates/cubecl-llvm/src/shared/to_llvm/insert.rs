use crate::prelude::*;

pub fn insert<O: Op + OneResultInterface>(
    ctx: &mut Context,
    inserter: &mut impl Inserter,
    op: &O,
) -> Value {
    inserter.insert_op(ctx, op);
    op.get_result(ctx)
}
