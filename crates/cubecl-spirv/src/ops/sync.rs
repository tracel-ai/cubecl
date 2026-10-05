use cubecl_ir::{
    dialect::synchronization::{SyncOp, SyncScope},
    prelude::*,
};
use pliron_spirv::ops::ControlBarrierOp;
use rspirv::spirv::{Capability, MemorySemantics, Scope};

use crate::{CustomCapabilitiesOp, ops::to_spirv_dialect::ToSpirvDialectOp};

// A barrier at the device's memory scope needs the capability the Vulkan memory model gates that
// scope behind, as a device-scope atomic does; nothing declares it for a kernel with no such atomic.
#[op_interface_impl]
impl CustomCapabilitiesOp for ControlBarrierOp {
    fn custom_capabilities(&self, ctx: &Context) -> Vec<Capability> {
        match self.get_attr_memory(ctx).0 {
            Scope::Device => vec![Capability::VulkanMemoryModelDeviceScopeKHR],
            _ => vec![],
        }
    }
}

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
            SyncScope::Unit => {
                rewriter.erase_operation(ctx, self.get_operation());
                return Ok(());
            }
        };
        let semantics = match scope {
            SyncScope::Plane => MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::SUBGROUP_MEMORY,
            SyncScope::Cube => MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY,
            // Under the Vulkan memory model a barrier orders, but only availability and visibility
            // operations carry one workgroup's plain writes to another at device scope.
            SyncScope::Device => {
                MemorySemantics::ACQUIRE_RELEASE
                    | MemorySemantics::UNIFORM_MEMORY
                    | MemorySemantics::WORKGROUP_MEMORY
                    | MemorySemantics::MAKE_AVAILABLE
                    | MemorySemantics::MAKE_VISIBLE
            }
            SyncScope::Unit => unreachable!(),
        };

        let sync = ControlBarrierOp::new(ctx, scope_exec, scope_mem, semantics);
        rewriter.append_op(ctx, &sync);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}
