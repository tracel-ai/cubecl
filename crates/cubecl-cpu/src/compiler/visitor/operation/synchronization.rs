use cubecl_core::ir::{Builtin, Synchronization};
use tracel_llvm::melior::{dialect::func, ir::attribute::FlatSymbolRefAttribute};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_synchronization(&mut self, synchronization: &Synchronization) {
        match synchronization {
            Synchronization::SyncCube => {
                let func_name = FlatSymbolRefAttribute::new(self.context, "sync_cube");
                let cube_dim = self.args_manager.get(Builtin::CubeDim);
                self.block.append_operation(func::call(
                    self.context,
                    func_name,
                    &[cube_dim],
                    &[],
                    self.location,
                ));
            }
            Synchronization::SyncPlane => {} //NOOP plane size is 1 on CPU
            Synchronization::SyncStorage => {
                panic!("SyncStorage is not supported")
            }
            Synchronization::SyncProxyShared => {
                panic!("SyncProxyShared is not supported")
            }
        }
    }
}
