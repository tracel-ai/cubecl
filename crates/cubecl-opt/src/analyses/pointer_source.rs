use core::{cell::RefCell, ops::Deref};
use std::collections::HashMap;

use cubecl_ir::{Memory, Operation, Variable};

use crate::{
    Function, GlobalState,
    analyses::{Analysis, post_order::PostOrder},
};

#[derive(Debug)]
pub struct PointerSource {
    /// The source variable of each pointer, propagated through copies
    pointer_sources: RefCell<HashMap<Variable, Variable>>,
}

impl Deref for PointerSource {
    type Target = RefCell<HashMap<Variable, Variable>>;

    fn deref(&self) -> &Self::Target {
        &self.pointer_sources
    }
}

impl PointerSource {
    pub fn new(opt: &mut Function, state: &GlobalState) -> Self {
        let blocks = opt.analysis::<PostOrder>(state).reverse();
        let mut pointer_sources = HashMap::new();
        for block in blocks {
            let insts = opt[block].ops.borrow().clone();
            let insts = insts.values();
            for inst in insts.filter(|it| it.out.is_some_and(|it| it.ty.is_ptr())) {
                let Some(out) = inst.out else {
                    unreachable!();
                };
                match &inst.operation {
                    Operation::Copy(variable) => {
                        let source = pointer_sources[variable];
                        pointer_sources.insert(out, source);
                    }
                    Operation::Memory(Memory::Reference(variable)) => {
                        pointer_sources.insert(out, *variable);
                    }
                    Operation::Memory(Memory::Index(op)) => {
                        pointer_sources.insert(out, op.list);
                    }
                    _ => {}
                }
            }
        }
        PointerSource {
            pointer_sources: RefCell::new(pointer_sources),
        }
    }
}

impl Analysis for PointerSource {
    fn init(opt: &mut crate::Function, state: &GlobalState) -> Self {
        PointerSource::new(opt, state)
    }
}
