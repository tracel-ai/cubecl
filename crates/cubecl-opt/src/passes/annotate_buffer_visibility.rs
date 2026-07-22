use alloc::boxed::Box;

use cubecl_ir::{
    attributes::{
        ATTR_BUFFER_BINDING, ATTR_BUFFER_IO, BufferBindingAttr, BufferIOAttr, FuncInterface,
    },
    prelude::*,
};
use pliron::builtin::ops::FuncOp;

use crate::analyses::pointer_source::GlobalVisibility;

/// Annotate global buffers with the correct visibility. Should be run on a module scope,
/// so non-inlined functions can contribute to the analysis and be annotated.
pub struct AnnotateGlobalVisibilityPass;

#[pass_name]
impl Pass for AnnotateGlobalVisibilityPass {
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
                    let buffer_binding = func
                        .get_arg_attr::<BufferBindingAttr>(ctx, i, &ATTR_BUFFER_BINDING)
                        .map(|it| *it);
                    if let Some(buffer) = buffer_binding
                        && let Some(visibility) =
                            global_visibility.visibility.get(&buffer.buffer_pos)
                    {
                        let io = match (visibility.readable, visibility.writable) {
                            (true, true) => BufferIOAttr::ReadWrite,
                            (true, false) => BufferIOAttr::ReadOnly,
                            (false, true) => BufferIOAttr::WriteOnly,
                            (false, false) => BufferIOAttr::Dead,
                        };
                        func.set_arg_attr(ctx, i, &ATTR_BUFFER_IO, Box::new(io));
                    }
                }
            },
        );
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}
