use core::any::type_name;

use cubecl_ir::{
    attributes::{
        ATTR_BUFFER_BINDING, ATTR_READ_WRITE, ATTR_READONLY, ATTR_WRITEONLY, BufferBindingAttr,
        FuncInterface,
    },
    prelude::*,
};
use pliron::builtin::ops::FuncOp;

use crate::analyses::pointer_source::GlobalVisibility;

/// Annotate global buffers with the correct visibility. Should be run on a module scope,
/// so non-inlined functions can contribute to the analysis and be annotated.
pub struct AnnotateGlobalVisibilityPass;

impl Pass for AnnotateGlobalVisibilityPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let global_visibility = analyses.get_analysis::<GlobalVisibility>(op, ctx)?;

        visit_all_ops_of_type::<FuncOp, _>(
            ctx,
            &mut &global_visibility,
            op,
            |ctx, global_visibility, func| {
                let num_args = func.get_entry_block(ctx).deref(ctx).get_num_arguments();
                for i in 0..num_args {
                    if let Some(buffer) = func
                        .get_arg_attr(ctx, i, &ATTR_BUFFER_BINDING)
                        .map(|it| it.clone())
                        && let Ok(buffer) = buffer.downcast::<BufferBindingAttr>()
                        && let Some(visibility) =
                            global_visibility.visibility.get(&buffer.buffer_pos)
                    {
                        match (visibility.readable, visibility.writable) {
                            (true, true) => func.set_arg_attr_unit(ctx, i, &ATTR_READ_WRITE),
                            (true, false) => func.set_arg_attr_unit(ctx, i, &ATTR_READONLY),
                            (false, true) => func.set_arg_attr_unit(ctx, i, &ATTR_WRITEONLY),
                            _ => {}
                        }
                    }
                }
            },
        );
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}
