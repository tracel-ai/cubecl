use core::{cell::RefCell, ops::Deref};

use cubecl_ir::{Memory, Operation, Variable};
use hashbrown::HashMap;

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
    pub fn new(func: &mut Function, state: &GlobalState) -> Self {
        let blocks = func.analysis::<PostOrder>(state).reverse();
        let mut pointer_sources = HashMap::new();
        for block in blocks {
            let insts = func[block].ops.borrow().clone();
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

    pub fn get(&self, var: &Variable) -> Option<Variable> {
        self.borrow().get(var).copied()
    }
}

impl Analysis for PointerSource {
    fn init(opt: &mut crate::Function, state: &GlobalState) -> Self {
        PointerSource::new(opt, state)
    }
}
