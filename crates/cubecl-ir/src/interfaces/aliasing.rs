use pliron::{basic_block::BasicBlock, value::DefiningEntity};
use smallvec::SmallVec;

use crate::prelude::*;

#[op_interface]
pub trait AliasingOp: OneResultInterface {
    verify_op_succ!();
    fn source_ptr(&self, ctx: &Context) -> SmallVec<[Value; 4]>;
    fn dest_ptr(&self, ctx: &Context) -> SmallVec<[Value; 4]>;
}

pub trait PointerExt {
    fn find_root_index(&self, ctx: &Context) -> usize;
    fn get_root_ptr_defining_entity(&self, ctx: &Context) -> DefiningEntity;
    fn get_root_ptr_defining_op(&self, ctx: &Context) -> Option<Ptr<Operation>> {
        match self.get_root_ptr_defining_entity(ctx) {
            DefiningEntity::Op(ptr) => Some(ptr),
            DefiningEntity::Block(_) => None,
        }
    }
    fn get_root_ptr_defining_block(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        match self.get_root_ptr_defining_entity(ctx) {
            DefiningEntity::Op(_) => None,
            DefiningEntity::Block(ptr) => Some(ptr),
        }
    }
    fn get_root_ptr(&self, ctx: &Context) -> Value {
        let index = self.find_root_index(ctx);
        match self.get_root_ptr_defining_entity(ctx) {
            DefiningEntity::Op(op) => op.deref(ctx).get_result(index),
            DefiningEntity::Block(block) => block.deref(ctx).get_argument(index),
        }
    }
}

impl PointerExt for Value {
    fn get_root_ptr_defining_entity(&self, ctx: &Context) -> DefiningEntity {
        match self.defining_entity() {
            DefiningEntity::Op(op) => {
                if let Some(aliasing) = op_cast::<dyn AliasingOp>(&*op.dyn_op(ctx))
                    && let &[source] = aliasing.source_ptr(ctx).as_slice()
                {
                    source.get_root_ptr_defining_entity(ctx)
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
                    && let &[source] = aliasing.source_ptr(ctx).as_slice() =>
            {
                source.find_root_index(ctx)
            }
            _ => self.find_index(ctx),
        }
    }
}
