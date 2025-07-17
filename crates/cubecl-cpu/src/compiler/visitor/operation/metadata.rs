use cubecl_core::ir::Metadata;
use tracel_llvm::melior::dialect::memref;

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    fn append_metadata(&mut self, offset: u32, out: Variable) {
        let metadata_memref = self.args_manager.metadata_memref.unwrap();
        let offset = self
            .block
            .const_int_from_type(
                self.context,
                self.location,
                offset as i64,
                Type::index(self.context),
            )
            .unwrap();
        let result = self.append_operation_with_result(memref::load(
            metadata_memref,
            &[offset],
            self.location,
        ));
        self.insert_variable(out, result);
    }

    pub fn visit_metadata(&mut self, metadata: &Metadata, out: Variable) {
        match metadata {
            Metadata::Length { var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.len_index(position);
                self.append_metadata(offset, out);
            }
            Metadata::BufferLength { var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.buffer_len_index(position);
                self.append_metadata(offset, out);
            }
            _ => todo!("This metadata is not yet implemented {}", metadata),
        }
    }
}
