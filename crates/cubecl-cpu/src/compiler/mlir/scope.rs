use cubecl_core::ir::Scope;
use melior::{
    dialect::{
        arith,
        llvm::{self, AllocaOptions},
    },
    ir::{BlockLike, Type, attribute::IntegerAttribute},
};

use crate::compiler::mlir::item::ElemSize;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_scope(&self, scope: &Scope) {
        for instruction in scope.instructions.iter().take(1) {
            println!("{:#?}", instruction);
            println!("{}", instruction);
            match instruction.out {
                Some(out) => {
                    let array_size = out.item.size() as i64;
                    let array_size = self
                        .block
                        .append_operation(arith::constant(
                            self.context,
                            IntegerAttribute::new(Type::index(self.context), array_size).into(),
                            self.location,
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    let ptr_type = self.elem_to_type(out.elem());
                    let out = self
                        .block
                        .append_operation(llvm::alloca(
                            self.context,
                            array_size,
                            ptr_type,
                            self.location,
                            AllocaOptions::new(),
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    self.visit_operation_with_out(&instruction.operation, out);
                }
                _ => {
                    todo!("Implement operation without out");
                }
            }
        }
    }
}
