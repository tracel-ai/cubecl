use cubecl_core::ir::attributes::EntrypointInterface;
use cubecl_core::ir::dialect::branch::YieldOp;
use cubecl_core::ir::prelude::*;
use pliron::basic_block::BasicBlock;
use pliron::builtin::ops::FuncOp;
use pliron::linked_list::ContainsLinkedList;
use pliron::printable::Printable;

#[derive(Default)]
pub struct InsertConstantEmulationPass;

#[pass_name]
impl Pass for InsertConstantEmulationPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let Some(func) = op.as_op::<FuncOp>(ctx) else {
            return Ok(res);
        };
        if func.get_entrypoint_abi(ctx).is_none() {
            return Ok(res);
        }
        println!("{}", func.get_type(ctx).disp(ctx));
        // let entry_block = func.get_entry_block(ctx);

        res.ir_changed = IRStatus::Unchanged;
        Ok(res)
    }
}
