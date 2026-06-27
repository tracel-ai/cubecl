use core::any::type_name;

use ::pliron::context::Context;
use cubecl_ir::{
    AddressSpace,
    attributes::IndexAttr,
    prelude::*,
    types::{BytesType, PointerType},
};
use pliron::{builtin::ops::FuncOp, irbuild::listener::DummyListener};

use crate::SharedLiveness;

#[cube_op(
    name = "cube.alloc_shared",
    format = "`size = ` attr($size, $IndexAttr) `, align = ` attr($alignment, $IndexAttr)"
)]
#[result_ty(fixed = PointerType::get(ctx, BytesType::get(ctx).into(), AddressSpace::Shared).to_handle())]
pub struct AllocSharedOp {
    pub size: IndexAttr,
    pub alignment: IndexAttr,
}

#[cube_op(
    name = "cube.slice_shared",
    format = "$0 `[` attr($offset, $IndexAttr) `] : ` type($0)"
)]
#[result_ty(argument)]
pub struct SliceSharedOp {
    pub block: Value,
    pub offset: IndexAttr,
}

/// Allocates shared memory as a single block and attaches offsets to shared memory declarations.
pub struct AllocateSharedMemoryBlockPass;

impl Pass for AllocateSharedMemoryBlockPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let analysis = analyses.get_analysis::<SharedLiveness>(op, ctx)?;
        let mut rewriter = IRRewriter::<DummyListener>::default();
        let op = op.dyn_op(ctx).downcast::<FuncOp>().unwrap();

        let allocs = analysis.allocations.values().copied().collect::<Vec<_>>();

        if !analysis.allocations.is_empty() {
            let size = allocs.iter().map(|it| it.end(ctx)).max().unwrap();
            let alignment = allocs.iter().map(|it| it.smem.alignment).max().unwrap();

            let entry = op.get_entry_block(ctx);
            let alloc = AllocSharedOp::new(ctx, size, alignment);
            alloc.get_operation().insert_at_front(entry, ctx);
            res.ir_changed |= IRStatus::Changed;
            let block = alloc.get_result(ctx);

            for alloc in allocs {
                let declaration = alloc.value.defining_op().expect("Should be op");
                let ptr_ty = declaration.result(ctx).get_type(ctx);
                let slice = SliceSharedOp::new(ctx, ptr_ty, block, alloc.offset);
                slice.get_operation().insert_before(ctx, declaration);
                rewriter.replace_operation(ctx, declaration, slice.get_operation());
            }
        }

        Ok(res)
    }
}
