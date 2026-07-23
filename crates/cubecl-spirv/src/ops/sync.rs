use cubecl_ir::{
    dialect::synchronization::{SyncOp, SyncScope},
    prelude::*,
};
use pliron_spirv::ops::ControlBarrierOp;
use rspirv::spirv::{MemorySemantics, Scope};

use crate::ops::to_spirv_dialect::ToSpirvDialectOp;

#[op_interface_impl]
impl ToSpirvDialectOp for SyncOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let scope = self.scope(ctx).0;
        let (scope_exec, scope_mem) = match scope {
            SyncScope::Plane => (Scope::Subgroup, Scope::Subgroup),
            SyncScope::Cube => (Scope::Workgroup, Scope::Workgroup),
            SyncScope::Device => (Scope::Workgroup, Scope::Device),
        };
        let semantics = match scope {
            SyncScope::Plane => MemorySemantics::RELAXED,
            SyncScope::Cube => MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY,
            SyncScope::Device => MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::UNIFORM_MEMORY,
        };

        let sync = ControlBarrierOp::new(ctx, scope_exec, scope_mem, semantics);
        rewriter.append_op(ctx, &sync);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}
