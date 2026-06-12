use cubecl_core::ir::{self as cube, Metadata};
use tracel_llvm::mlir_rs::dialect::{arith, index, memref};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    fn append_metadata(&mut self, offset: u32, out: cube::ExpandValue) {
        let metadata_memref = self.args_manager.static_metadata_memref.unwrap();
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
        self.insert_value(out, result);
    }

    fn append_extended_metadata(
        &mut self,
        offset: u32,
        dim: cube::ExpandValue,
        out: cube::ExpandValue,
    ) {
        let static_metadata_memref = self.args_manager.static_metadata_memref.unwrap();
        let dynamic_metadata_memref = self.args_manager.dynamic_metadata_memref.unwrap();
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
            static_metadata_memref,
            &[offset],
            self.location,
        ));
        let first_rank = self.append_operation_with_result(index::casts(
            first_rank,
            Type::index(self.context),
            self.location,
        ));

        let dim = self.get_index(dim, dim.ty, true);
        let offset = self.append_operation_with_result(arith::addi(first_rank, dim, self.location));
        let result = self.append_operation_with_result(memref::load(
            dynamic_metadata_memref,
            &[offset],
            self.location,
        ));
        self.insert_value(out, result);
    }

    pub fn visit_metadata(&mut self, metadata: &Metadata, out: cube::ExpandValue) {
        match metadata {
            Metadata::BufferLength { list } => {
                let position = self.args_manager.buffer_position(list);
                let offset = self.args_manager.metadata.buffer_len_index(position);
                self.append_metadata(offset, out);
            }
            Metadata::Shape { dim, list } => {
                let position = self.args_manager.ext_meta_position(list);
                let offset = self.args_manager.metadata.shape_offset_index(position);
                self.append_extended_metadata(offset, *dim, out);
            }
            Metadata::Stride { dim, list } => {
                let position = self.args_manager.ext_meta_position(list);
                let offset = self.args_manager.metadata.stride_offset_index(position);
                self.append_extended_metadata(offset, *dim, out);
            }
        }
    }
}
