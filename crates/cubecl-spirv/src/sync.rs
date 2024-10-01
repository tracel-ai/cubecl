use cubecl_core::ir::Synchronization;
use rspirv::spirv::{MemorySemantics, Scope};

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_sync(&mut self, sync: Synchronization) {
        match sync {
            Synchronization::SyncUnits => {
                let scope = self.const_u32(Scope::Workgroup as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE
                    | MemorySemantics::WORKGROUP_MEMORY
                    | MemorySemantics::SUBGROUP_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope, scope, semantics).unwrap();
            }
            Synchronization::SyncStorage => {
                let scope = self.const_u32(Scope::Device as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE
                    | MemorySemantics::WORKGROUP_MEMORY
                    | MemorySemantics::SUBGROUP_MEMORY
                    | MemorySemantics::CROSS_WORKGROUP_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.memory_barrier(scope, semantics).unwrap();
            }
        }
    }
}
