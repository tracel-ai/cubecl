use cubecl_core::ir::Synchronization;
use rspirv::spirv::{MemorySemantics, Scope};

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_sync(&mut self, sync: Synchronization) {
        match sync {
            Synchronization::SyncCube => {
                // Adopting wgpu semantics
                let scope = self.const_u32(Scope::Workgroup as u32);
                let semantics =
                    MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope, scope, semantics).unwrap();
            }
            Synchronization::SyncPlane => {
                // Adopting wgpu semantics
                let scope = self.const_u32(Scope::Subgroup as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::SUBGROUP_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope, scope, semantics).unwrap();
            }
            Synchronization::SyncStorage => {
                // Adopting wgpu semantics
                let scope_exec = self.const_u32(Scope::Workgroup as u32);
                let scope_mem = self.const_u32(Scope::Device as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::UNIFORM_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope_exec, scope_mem, semantics)
                    .unwrap();
            }
            Synchronization::SyncProxyShared => panic!("TMA proxy sync not supported in SPIR-V"),
        }
    }
}
