use core::{cell::RefCell, ops::Deref};

use cubecl_ir::{Id, Memory, Operation, Value, ValueKind};
use hashbrown::HashMap;

use crate::{
    Function, GlobalState, MemoryBlock,
    analyses::{Analysis, post_order::PostOrder},
};

#[derive(Debug)]
pub struct PointerSource {
    /// The source memory of each pointer, propagated through copies
    pointer_sources: RefCell<HashMap<Id, MemoryBlock>>,
}

impl Deref for PointerSource {
    type Target = RefCell<HashMap<Id, MemoryBlock>>;

    fn deref(&self) -> &Self::Target {
        &self.pointer_sources
    }
}

impl PointerSource {
    pub fn new(func: &mut Function, state: &GlobalState) -> Self {
        let blocks = func.analysis::<PostOrder>(state).reverse();
        let mut pointer_sources = HashMap::new();

        for (id, mem) in func.memories.iter() {
            pointer_sources.insert(*id, *mem);
        }

        for block in blocks {
            let insts = func[block].ops.borrow().clone();
            let insts = insts.values();
            for inst in insts.filter(|it| it.out.is_some_and(|it| it.ty.is_ptr())) {
                let Some(out) = inst.out else {
                    unreachable!();
                };
                match &inst.operation {
                    Operation::Copy(Value {
                        kind: ValueKind::Value { id },
                        ..
                    }) => {
                        if let Some(source) = pointer_sources.get(id) {
                            pointer_sources.insert(out.id(), *source);
                        }
                    }
                    Operation::Memory(Memory::Index(op)) => {
                        if let Some(mem) = pointer_sources.get(&op.list.id()) {
                            pointer_sources.insert(out.id(), *mem);
                        }
                    }
                    _ => {}
                }
            }
        }
        PointerSource {
            pointer_sources: RefCell::new(pointer_sources),
        }
    }

    pub fn get(&self, val: &Value) -> Option<MemoryBlock> {
        if let ValueKind::Value { id } = &val.kind {
            self.borrow().get(id).copied()
        } else {
            None
        }
    }
}

impl Analysis for PointerSource {
    fn init(opt: &mut crate::Function, state: &GlobalState) -> Self {
        PointerSource::new(opt, state)
    }
}
