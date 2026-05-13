use alloc::{rc::Rc, vec, vec::Vec};
use core::cell::{RefCell, RefMut};
use cubecl_runtime::kernel::Visibility;
use derive_more::{Deref, DerefMut};

use cubecl_ir::{GlobalState, Id, Instruction, Memory, Operation, Scope, Variable, VariableKind};
use hashbrown::{HashMap, HashSet};

use crate::post_processing::{
    util::AtomicCounter,
    visitor::{InstructionVisitor, Visitor},
};

#[derive(Debug, Clone, Default)]
pub struct GlobalAnalyses {
    ptr_source: Rc<RefCell<PointerSource>>,
    used_values: Rc<RefCell<UsedValues>>,
}

impl GlobalAnalyses {
    pub fn recalculate_pointer_source(&self, scope: &Scope) {
        *self.ptr_source.borrow_mut() = PointerSource::new(scope);
    }

    pub fn recalculate_used_values(&self, scope: &Scope) {
        let mut used_values = UsedValues::default();
        used_values.visit_scope(scope, self, &AtomicCounter::new(0));
        *self.used_values.borrow_mut() = used_values;
    }

    pub fn ptr_source(&self) -> RefMut<'_, PointerSource> {
        self.ptr_source.borrow_mut()
    }

    pub fn used_values(&self) -> RefMut<'_, UsedValues> {
        self.used_values.borrow_mut()
    }
}

#[derive(Default, Debug, Deref, DerefMut)]
pub struct UsedValues {
    used: HashSet<Variable>,
}

impl InstructionVisitor for UsedValues {
    fn visit_instruction(
        &mut self,
        mut inst: Instruction,
        _global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(self);
        visitor.visit_operation(&mut inst.operation, analyses, |this, var| {
            this.used.insert(*var);
        });
        vec![inst]
    }
}

#[derive(Debug, Default, Deref, DerefMut)]
pub struct PointerSource {
    sources: HashMap<VariableKind, Variable>,
}

impl PointerSource {
    pub fn new(scope: &Scope) -> Self {
        let mut this = PointerSource::default();
        this.visit_scope(scope, &GlobalAnalyses::default(), &AtomicCounter::new(0));
        this
    }
}

impl InstructionVisitor for PointerSource {
    fn visit_instruction(
        &mut self,
        inst: Instruction,
        _global_state: &GlobalState,
        _analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        match &inst.operation {
            Operation::Copy(var) if var.ty.is_ptr() && inst.out().ty.is_ptr() => {
                let source = self.sources[&var.kind];
                self.sources.insert(inst.out().kind, source);
            }
            Operation::Memory(memory) => match memory {
                Memory::Reference(variable) => {
                    self.sources.insert(inst.out().kind, *variable);
                }
                Memory::Index(index_operands) => {
                    self.sources.insert(inst.out().kind, index_operands.list);
                }
                _ => {}
            },
            _ => {}
        }
        vec![inst]
    }
}

#[derive(Debug, Deref, Default)]
pub struct BufferVisibility {
    buffers: Vec<Visibility>,
}

impl From<BufferVisibility> for Vec<Visibility> {
    fn from(value: BufferVisibility) -> Self {
        value.buffers
    }
}

impl BufferVisibility {
    pub fn new(scope: &Scope, analyses: &GlobalAnalyses) -> Self {
        let mut this = BufferVisibility::default();
        this.visit_scope(scope, analyses, &AtomicCounter::new(0));
        this
    }
}

impl InstructionVisitor for BufferVisibility {
    fn visit_instruction(
        &mut self,
        mut inst: Instruction,
        _global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(self);

        visitor.visit_instruction(
            &mut inst,
            analyses,
            |this, var| {
                if let Some(id) = global_buffer_id(var) {
                    this.set_readable(id as usize);
                }
            },
            |this, var| {
                if let Some(id) = global_buffer_id(var) {
                    this.set_writable(id as usize);
                }
            },
        );

        vec![inst]
    }
}

impl BufferVisibility {
    // There for consistency and in case we make it more granular like it is in SPIR-V
    fn set_readable(&mut self, id: usize) {
        if self.buffers.len() <= id {
            self.buffers.resize(id + 1, Visibility::Read);
        }
    }

    fn set_writable(&mut self, id: usize) {
        if self.buffers.len() <= id {
            self.buffers.resize(id + 1, Visibility::Read);
        }
        self.buffers[id] = Visibility::ReadWrite;
    }
}

fn global_buffer_id(variable: &Variable) -> Option<Id> {
    match variable.kind {
        VariableKind::GlobalBuffer(id) | VariableKind::TensorMap(id) => Some(id),
        _ => None,
    }
}
