use cubecl_core::ir::Metadata;
use tracel_llvm::melior::{
    dialect::{arith, index, memref},
    ir::{attribute::IntegerAttribute, r#type::IntegerType},
};

use crate::compiler::mlir::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_metadata(&mut self, metadata: &Metadata, out: Variable) {
        match metadata {
            Metadata::Length { var } => {
                let constant = self.append_operation_with_result(arith::constant(
                    self.context,
                    IntegerAttribute::new(Type::index(self.context), 0).into(),
                    self.location,
                ));
                let variable = self.get_memory(*var);
                let value = self.append_operation_with_result(memref::dim(
                    variable,
                    constant,
                    self.location,
                ));
                let integer_type = IntegerType::new(self.context, 32);
                let value = self.append_operation_with_result(index::casts(
                    value,
                    integer_type.into(),
                    self.location,
                ));
                self.insert_variable(out, value);
            }
            Metadata::BufferLength { var } => {
                let constant = self.append_operation_with_result(arith::constant(
                    self.context,
                    IntegerAttribute::new(Type::index(self.context), 0).into(),
                    self.location,
                ));
                let variable = self.get_memory(*var);
                let value = self.append_operation_with_result(memref::dim(
                    variable,
                    constant,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
            _ => todo!("This metadata is not yet implemented {}", metadata),
        }
    }
}
