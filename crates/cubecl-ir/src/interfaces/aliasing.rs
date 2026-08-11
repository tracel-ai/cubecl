use pliron::{basic_block::BasicBlock, value::DefiningEntity};

use crate::prelude::*;

#[op_interface]
pub trait AliasingOp: OneResultInterface {
    verify_op_succ!();
    fn source_ptr(&self, ctx: &Context) -> Option<Value>;
}

pub trait PointerExt {
    fn find_root_index(&self, ctx: &Context) -> usize;
    fn get_root_defining_entity(&self, ctx: &Context) -> DefiningEntity;
    fn get_root_defining_op(&self, ctx: &Context) -> Option<Ptr<Operation>> {
        match self.get_root_defining_entity(ctx) {
            DefiningEntity::Op(ptr) => Some(ptr),
            DefiningEntity::Block(_) => None,
        }
    }
    fn get_root_defining_block(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        match self.get_root_defining_entity(ctx) {
            DefiningEntity::Op(_) => None,
            DefiningEntity::Block(ptr) => Some(ptr),
        }
    }
}

impl PointerExt for Value {
    fn get_root_defining_entity(&self, ctx: &Context) -> DefiningEntity {
        match self.defining_entity() {
            DefiningEntity::Op(op) => {
                if let Some(aliasing) = op_cast::<dyn AliasingOp>(&*op.dyn_op(ctx))
                    && let Some(source) = aliasing.source_ptr(ctx)
                {
                    source.get_root_defining_entity(ctx)
                } else {
                    DefiningEntity::Op(op)
                }
            }
            block @ DefiningEntity::Block(_) => block,
        }
    }

    fn find_root_index(&self, ctx: &Context) -> usize {
        match self.defining_entity() {
            DefiningEntity::Op(op)
                if let Some(aliasing) = op_cast::<dyn AliasingOp>(&*op.dyn_op(ctx))
                    && let Some(source) = aliasing.source_ptr(ctx) =>
            {
                source.find_root_index(ctx)
            }
            _ => self.find_index(ctx),
        }
    }
}
