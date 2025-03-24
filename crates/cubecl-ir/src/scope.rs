use alloc::{borrow::Cow, rc::Rc, vec::Vec};
use core::{any::TypeId, cell::RefCell};
use hashbrown::{HashMap, HashSet};

use crate::{BarrierLevel, CubeFnSource, ExpandElement, Matrix, SourceLoc, TypeHash};

use super::{
    Allocator, Elem, Id, Instruction, Item, Operation, UIntKind, Variable, VariableKind,
    processing::ScopeProcessing,
};

/// The scope is the main [operation](Operation) and [variable](Variable) container that simplify
/// the process of reading inputs, creating local variables and adding new operations.
///
/// Notes:
///
/// This type isn't responsible for creating [shader bindings](super::Binding) and figuring out which
/// variable can be written to.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
#[allow(missing_docs)]
pub struct Scope {
    pub depth: u8,
    pub instructions: Vec<Instruction>,
    pub locals: Vec<Variable>,
    matrices: Vec<Variable>,
    pipelines: Vec<Variable>,
    barriers: Vec<Variable>,
    slices: Vec<Variable>,
    shared_memories: Vec<Variable>,
    pub const_arrays: Vec<(Variable, Vec<Variable>)>,
    local_arrays: Vec<Variable>,
    reads_global: Vec<(Variable, ReadingStrategy, Variable, Variable)>,
    index_offset_with_output_layout_position: Vec<usize>,
    writes_global: Vec<(Variable, Variable, Variable)>,
    reads_scalar: Vec<(Variable, Variable)>,
    pub layout_ref: Option<Variable>,
    pub allocator: Allocator,
    pub debug: DebugInfo,
    #[type_hash(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub typemap: Rc<RefCell<HashMap<TypeId, Elem>>>,
}

/// Debug related fields, most of these are global
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct DebugInfo {
    pub enabled: bool,
    pub sources: Rc<RefCell<HashSet<CubeFnSource>>>,
    pub variable_names: Rc<RefCell<HashMap<Variable, Cow<'static, str>>>>,
    pub source_loc: Option<SourceLoc>,
    pub entry_loc: Option<SourceLoc>,
}

impl core::hash::Hash for Scope {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        self.depth.hash(ra_expand_state);
        self.instructions.hash(ra_expand_state);
        self.locals.hash(ra_expand_state);
        self.matrices.hash(ra_expand_state);
        self.pipelines.hash(ra_expand_state);
        self.barriers.hash(ra_expand_state);
        self.slices.hash(ra_expand_state);
        self.shared_memories.hash(ra_expand_state);
        self.const_arrays.hash(ra_expand_state);
        self.local_arrays.hash(ra_expand_state);
        self.reads_global.hash(ra_expand_state);
        self.index_offset_with_output_layout_position
            .hash(ra_expand_state);
        self.writes_global.hash(ra_expand_state);
        self.reads_scalar.hash(ra_expand_state);
        self.layout_ref.hash(ra_expand_state);
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash)]
#[allow(missing_docs)]
pub enum ReadingStrategy {
    /// Each element will be read in a way to be compatible with the output layout.
    OutputLayout,
    /// Keep the current layout.
    Plain,
}

impl Scope {
    /// Create a scope that is at the root of a
    /// [kernel definition](crate::ir::KernelDefinition).
    ///
    /// A local scope can be created with the [child](Self::child) method.
    pub fn root(debug_enabled: bool) -> Self {
        Self {
            depth: 0,
            instructions: Vec::new(),
            locals: Vec::new(),
            matrices: Vec::new(),
            pipelines: Vec::new(),
            barriers: Vec::new(),
            slices: Vec::new(),
            local_arrays: Vec::new(),
            shared_memories: Vec::new(),
            const_arrays: Vec::new(),
            reads_global: Vec::new(),
            index_offset_with_output_layout_position: Vec::new(),
            writes_global: Vec::new(),
            reads_scalar: Vec::new(),
            layout_ref: None,
            allocator: Allocator::default(),
            debug: DebugInfo {
                enabled: debug_enabled,
                sources: Default::default(),
                variable_names: Default::default(),
                source_loc: None,
                entry_loc: None,
            },
            typemap: Default::default(),
        }
    }

    /// Create a new matrix element.
    pub fn create_matrix(&mut self, matrix: Matrix) -> ExpandElement {
        let matrix = self.allocator.create_matrix(matrix);
        self.add_matrix(*matrix);
        matrix
    }

    pub fn add_matrix(&mut self, variable: Variable) {
        self.matrices.push(variable);
    }

    /// Create a new pipeline element.
    pub fn create_pipeline(&mut self, item: Item, num_stages: u8) -> ExpandElement {
        let pipeline = self.allocator.create_pipeline(item, num_stages);
        self.add_pipeline(*pipeline);
        pipeline
    }

    /// Create a new barrier element.
    pub fn create_barrier(&mut self, item: Item, level: BarrierLevel) -> ExpandElement {
        let barrier = self.allocator.create_barrier(item, level);
        self.add_barrier(*barrier);
        barrier
    }

    pub fn add_pipeline(&mut self, variable: Variable) {
        self.pipelines.push(variable);
    }

    pub fn add_barrier(&mut self, variable: Variable) {
        self.barriers.push(variable);
    }

    /// Create a new slice element.
    pub fn create_slice(&mut self, item: Item) -> ExpandElement {
        let slice = self.allocator.create_slice(item);
        self.add_slice(*slice);
        slice
    }

    pub fn add_slice(&mut self, slice: Variable) {
        self.slices.push(slice);
    }

    /// Create a mutable variable of the given [item type](Item).
    pub fn create_local_mut<I: Into<Item>>(&mut self, item: I) -> ExpandElement {
        self.allocator.create_local_mut(item.into())
    }

    /// Create a mutable variable of the given [item type](Item).
    pub fn add_local_mut(&mut self, var: Variable) {
        if !self.locals.contains(&var) {
            self.locals.push(var);
        }
    }

    /// Create a new restricted variable. The variable is
    /// Useful for _for loops_ and other algorithms that require the control over initialization.
    pub fn create_local_restricted(&mut self, item: Item) -> ExpandElement {
        self.allocator.create_local_restricted(item)
    }

    /// Create a new immutable variable.
    pub fn create_local(&mut self, item: Item) -> ExpandElement {
        self.allocator.create_local(item)
    }

    /// Reads an input array to a local variable.
    ///
    /// The index refers to the argument position of the array in the compute shader.
    pub fn read_array<I: Into<Item>>(
        &mut self,
        index: Id,
        item: I,
        position: Variable,
    ) -> Variable {
        self.read_input_strategy(index, item.into(), ReadingStrategy::OutputLayout, position)
    }

    /// Reads an input scalar to a local variable.
    ///
    /// The index refers to the scalar position for the same [element](Elem) type.
    pub fn read_scalar(&mut self, index: Id, elem: Elem) -> Variable {
        let id = self.new_local_index();
        let local = Variable::new(VariableKind::LocalConst { id }, Item::new(elem));
        let scalar = Variable::new(VariableKind::GlobalScalar(index), Item::new(elem));

        self.reads_scalar.push((local, scalar));

        local
    }

    /// Retrieve the last local variable that was created.
    pub fn last_local_index(&self) -> Option<&Variable> {
        self.locals.last()
    }

    /// Writes a variable to given output.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub fn write_global(&mut self, input: Variable, output: Variable, position: Variable) {
        // This assumes that all outputs have the same layout
        if self.layout_ref.is_none() {
            self.layout_ref = Some(output);
        }
        self.writes_global.push((input, output, position));
    }

    /// Writes a variable to given output.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub fn write_global_custom(&mut self, output: Variable) {
        // This assumes that all outputs have the same layout
        if self.layout_ref.is_none() {
            self.layout_ref = Some(output);
        }
    }

    /// Update the [reading strategy](ReadingStrategy) for an input array.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub fn update_read(&mut self, index: Id, strategy: ReadingStrategy) {
        if let Some((_, strategy_old, _, _position)) = self
            .reads_global
            .iter_mut()
            .find(|(var, _, _, _)| var.index() == Some(index))
        {
            *strategy_old = strategy;
        }
    }

    #[allow(dead_code)]
    pub fn read_globals(&self) -> Vec<(Id, ReadingStrategy)> {
        self.reads_global
            .iter()
            .map(|(var, strategy, _, _)| match var.kind {
                VariableKind::GlobalInputArray(id) => (id, *strategy),
                _ => panic!("Can only read global input arrays."),
            })
            .collect()
    }

    /// Register an [operation](Operation) into the scope.
    pub fn register<T: Into<Instruction>>(&mut self, instruction: T) {
        let mut inst = instruction.into();
        inst.source_loc = self.debug.source_loc.clone();
        self.instructions.push(inst)
    }

    /// Resolve the element type of the given generic type.
    pub fn resolve_elem<T: 'static>(&self) -> Option<Elem> {
        let map = self.typemap.borrow();
        let result = map.get(&TypeId::of::<T>());

        result.cloned()
    }

    /// Register the element type for the given generic type.
    pub fn register_elem<T: 'static>(&mut self, elem: Elem) {
        let mut map = self.typemap.borrow_mut();

        map.insert(TypeId::of::<T>(), elem);
    }

    /// Create an empty child scope.
    pub fn child(&mut self) -> Self {
        Self {
            depth: self.depth + 1,
            instructions: Vec::new(),
            locals: Vec::new(),
            matrices: Vec::new(),
            pipelines: Vec::new(),
            barriers: Vec::new(),
            slices: Vec::new(),
            shared_memories: Vec::new(),
            const_arrays: Vec::new(),
            local_arrays: Vec::new(),
            reads_global: Vec::new(),
            index_offset_with_output_layout_position: Vec::new(),
            writes_global: Vec::new(),
            reads_scalar: Vec::new(),
            layout_ref: self.layout_ref,
            allocator: self.allocator.clone(),
            debug: self.debug.clone(),
            typemap: self.typemap.clone(),
        }
    }

    /// Returns the variables and operations to be declared and executed.
    ///
    /// Notes:
    ///
    /// New operations and variables can be created within the same scope without having name
    /// conflicts.
    pub fn process(&mut self) -> ScopeProcessing {
        let mut variables = core::mem::take(&mut self.locals);

        for var in self.matrices.drain(..) {
            variables.push(var);
        }
        for var in self.slices.drain(..) {
            variables.push(var);
        }

        let mut instructions = Vec::new();

        for (local, scalar) in self.reads_scalar.drain(..) {
            instructions.push(Instruction::new(Operation::Copy(scalar), local));
            variables.push(local);
        }

        for inst in self.instructions.drain(..) {
            instructions.push(inst);
        }

        variables.extend(self.allocator.take_variables());

        ScopeProcessing {
            variables,
            instructions,
        }
        .optimize()
    }

    pub fn new_local_index(&self) -> u32 {
        self.allocator.new_local_index()
    }

    fn read_input_strategy(
        &mut self,
        index: Id,
        item: Item,
        strategy: ReadingStrategy,
        position: Variable,
    ) -> Variable {
        let item_global = match item.elem() {
            Elem::Bool => Item {
                elem: Elem::UInt(UIntKind::U32),
                vectorization: item.vectorization,
            },
            _ => item,
        };
        let input = Variable::new(VariableKind::GlobalInputArray(index), item_global);
        let id = self.new_local_index();
        let local = Variable::new(VariableKind::LocalMut { id }, item);
        self.reads_global.push((input, strategy, local, position));
        self.locals.push(local);
        local
    }

    /// Create a shared variable of the given [item type](Item).
    pub fn create_shared<I: Into<Item>>(
        &mut self,
        item: I,
        shared_memory_size: u32,
        alignment: Option<u32>,
    ) -> ExpandElement {
        let item = item.into();
        let index = self.new_local_index();
        let shared_memory = Variable::new(
            VariableKind::SharedMemory {
                id: index,
                length: shared_memory_size,
                alignment,
            },
            item,
        );
        self.shared_memories.push(shared_memory);
        ExpandElement::Plain(shared_memory)
    }

    /// Create a shared variable of the given [item type](Item).
    pub fn create_const_array<I: Into<Item>>(
        &mut self,
        item: I,
        data: Vec<Variable>,
    ) -> ExpandElement {
        let item = item.into();
        let index = self.new_local_index();
        let const_array = Variable::new(
            VariableKind::ConstantArray {
                id: index,
                length: data.len() as u32,
            },
            item,
        );
        self.const_arrays.push((const_array, data));
        ExpandElement::Plain(const_array)
    }

    /// Obtain the index-th input
    pub fn input(&mut self, id: Id, item: Item) -> ExpandElement {
        ExpandElement::Plain(crate::Variable::new(
            VariableKind::GlobalInputArray(id),
            item,
        ))
    }

    /// Obtain the index-th output
    pub fn output(&mut self, id: Id, item: Item) -> ExpandElement {
        let var = crate::Variable::new(VariableKind::GlobalOutputArray(id), item);
        self.write_global_custom(var);
        ExpandElement::Plain(var)
    }

    /// Obtain the index-th scalar
    pub fn scalar(&self, id: Id, elem: Elem) -> ExpandElement {
        ExpandElement::Plain(crate::Variable::new(
            VariableKind::GlobalScalar(id),
            Item::new(elem),
        ))
    }

    /// Create a local array of the given [item type](Item).
    pub fn create_local_array<I: Into<Item>>(&mut self, item: I, array_size: u32) -> ExpandElement {
        let local_array = self.allocator.create_local_array(item.into(), array_size);
        self.add_local_array(*local_array);
        local_array
    }

    pub fn add_local_array(&mut self, var: Variable) {
        self.local_arrays.push(var);
    }

    pub fn update_source(&mut self, source: CubeFnSource) {
        if self.debug.enabled {
            self.debug.sources.borrow_mut().insert(source.clone());
            self.debug.source_loc = Some(SourceLoc {
                line: source.line,
                column: source.column,
                source,
            });
            if self.debug.entry_loc.is_none() {
                self.debug.entry_loc = self.debug.source_loc.clone();
            }
        }
    }

    pub fn update_span(&mut self, line: u32, col: u32) {
        if let Some(loc) = self.debug.source_loc.as_mut() {
            loc.line = line;
            loc.column = col;
        }
    }

    pub fn update_variable_name(&self, variable: Variable, name: impl Into<Cow<'static, str>>) {
        if self.debug.enabled {
            self.debug
                .variable_names
                .borrow_mut()
                .insert(variable, name.into());
        }
    }
}
