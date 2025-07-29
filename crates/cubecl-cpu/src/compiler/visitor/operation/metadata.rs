use cubecl_core::ir::Metadata;
use tracel_llvm::melior::dialect::{arith, index, memref};

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

    fn append_extended_metadata(&mut self, offset: u32, dim: Variable, out: Variable) {
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
        let first_rank = self.append_operation_with_result(memref::load(
            metadata_memref,
            &[offset],
            self.location,
        ));
        let first_rank = self.append_operation_with_result(index::casts(
            first_rank,
            Type::index(self.context),
            self.location,
        ));

        let dim = self.get_index(dim, dim.item);
        let offset = self.append_operation_with_result(arith::addi(first_rank, dim, self.location));
        let result = self.append_operation_with_result(memref::load(
            metadata_memref,
            &[offset],
            self.location,
        ));
        self.insert_variable(out, result);
    }

    pub fn visit_metadata(&mut self, metadata: &Metadata, out: Variable) {
        match metadata {
            Metadata::BufferLength { var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.buffer_len_index(position);
                self.append_metadata(offset, out);
            }
            Metadata::Length { var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.len_index(position);
                self.append_metadata(offset, out);
            }
            Metadata::Rank { var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.rank_index(position);
                self.append_metadata(offset, out);
            }
            Metadata::Shape { dim, var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.shape_offset_index(position);
                self.append_extended_metadata(offset, *dim, out);
            }
            Metadata::Stride { dim, var } => {
                let position = self.args_manager.ext_meta_position(*var);
                let offset = self.args_manager.metadata.stride_offset_index(position);
                self.append_extended_metadata(offset, *dim, out);
            }
        }
    }
}
