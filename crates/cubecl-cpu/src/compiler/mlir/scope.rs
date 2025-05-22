use super::prelude::*;
use cubecl_core::ir::Scope;
use melior::{
    Context,
    dialect::{
        arith,
        llvm::{self, AllocaOptions},
    },
    ir::{Block, BlockLike, Location, Type, attribute::IntegerAttribute, r#type::IntegerType},
};

pub(super) trait VisitScope {
    fn visit<'a>(&self, block: &Block, context: &'a Context, location: Location);
}

impl VisitScope for Scope {
    fn visit<'a>(&self, block: &Block, context: &'a Context, location: Location) {
        for instruction in self.instructions.iter().take(1) {
            println!("{:#?}", instruction);
            println!("{}", instruction);
            match instruction.out {
                Some(out) => {
                    let array_size = out.item.size() as i64;
                    let array_size = block
                        .append_operation(arith::constant(
                            context,
                            IntegerAttribute::new(Type::index(context), array_size).into(),
                            location,
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    let ptr_type = out.elem().visit(context);
                    let out = block
                        .append_operation(llvm::alloca(
                            context,
                            array_size,
                            ptr_type,
                            location,
                            AllocaOptions::new(),
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    instruction
                        .operation
                        .visit_with_out(block, context, location, out);
                }
                _ => {
                    todo!("Implement operation without out");
                }
            }
        }
    }
}
