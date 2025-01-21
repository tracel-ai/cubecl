use type_hash::TypeHash;

use crate::ConstantScalarValue;

use super::{
    cpa, processing::ScopeProcessing, Allocator, Elem, Id, Instruction, Item, Operation, UIntKind,
    Variable, VariableKind,
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
    pub operations: Vec<Instruction>,
    pub locals: Vec<Variable>,
    matrices: Vec<Variable>,
    pipelines: Vec<Variable>,
    slices: Vec<Variable>,
    shared_memories: Vec<Variable>,
    pub const_arrays: Vec<(Variable, Vec<Variable>)>,
    local_arrays: Vec<Variable>,
    reads_global: Vec<(Variable, ReadingStrategy, Variable, Variable)>,
    index_offset_with_output_layout_position: Vec<usize>,
    writes_global: Vec<(Variable, Variable, Variable)>,
    reads_scalar: Vec<(Variable, Variable)>,
    pub layout_ref: Option<Variable>,
    #[cfg_attr(feature = "serde", serde(skip))]
    #[type_hash(skip)]
    pub allocator: Allocator,
}

impl core::hash::Hash for Scope {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        self.depth.hash(ra_expand_state);
        self.operations.hash(ra_expand_state);
        self.locals.hash(ra_expand_state);
        self.matrices.hash(ra_expand_state);
        self.pipelines.hash(ra_expand_state);
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
    pub fn root() -> Self {
        Self {
            depth: 0,
            operations: Vec::new(),
            locals: Vec::new(),
            matrices: Vec::new(),
            pipelines: Vec::new(),
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
        }
    }

    /// Create a variable initialized at zero.
    pub fn zero<I: Into<Item>>(&mut self, item: I) -> Variable {
        let local = self.create_local(item.into());
        let zero: Variable = 0u32.into();
        cpa!(self, local = zero);
        local
    }

    /// Create a variable initialized at some value.
    pub fn create_with_value<E, I>(&mut self, value: E, item: I) -> Variable
    where
        E: num_traits::ToPrimitive,
        I: Into<Item> + Copy,
    {
        let item: Item = item.into();
        let value = match item.elem() {
            Elem::Float(kind) | Elem::AtomicFloat(kind) => {
                ConstantScalarValue::Float(value.to_f64().unwrap(), kind)
            }
            Elem::Int(kind) | Elem::AtomicInt(kind) => {
                ConstantScalarValue::Int(value.to_i64().unwrap(), kind)
            }
            Elem::UInt(kind) | Elem::AtomicUInt(kind) => {
                ConstantScalarValue::UInt(value.to_u64().unwrap(), kind)
            }
            Elem::Bool => ConstantScalarValue::Bool(value.to_u32().unwrap() == 1),
        };
        let local = self.create_local(item);
        let value = Variable::constant(value);
        cpa!(self, local = value);
        local
    }

    pub fn add_matrix(&mut self, variable: Variable) {
        self.matrices.push(variable);
    }

    pub fn add_pipeline(&mut self, variable: Variable) {
        self.pipelines.push(variable);
    }

    pub fn add_slice(&mut self, slice: Variable) {
        self.slices.push(slice);
    }

    /// Create a mutable variable of the given [item type](Item).
    pub fn create_local_mut<I: Into<Item>>(&mut self, item: I) -> Variable {
        let id = self.new_local_index();
        let local = Variable::new(VariableKind::LocalMut { id }, item.into());
        self.add_local_mut(local);
        local
    }

    /// Create a mutable variable of the given [item type](Item).
    pub fn add_local_mut(&mut self, var: Variable) {
        if !self.locals.contains(&var) {
            self.locals.push(var);
        }
    }

    /// Create a new restricted variable. The variable is
    /// Useful for _for loops_ and other algorithms that require the control over initialization.
    pub fn create_local_restricted(&mut self, item: Item) -> Variable {
        *self.allocator.create_local_restricted(item)
    }

    /// Create a new immutable variable.
    pub fn create_local(&mut self, item: Item) -> Variable {
        *self.allocator.create_local(item)
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
    pub fn register<T: Into<Instruction>>(&mut self, operation: T) {
        self.operations.push(operation.into())
    }

    /// Create an empty child scope.
    pub fn child(&mut self) -> Self {
        Self {
            depth: self.depth + 1,
            operations: Vec::new(),
            locals: Vec::new(),
            matrices: Vec::new(),
            pipelines: Vec::new(),
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

        let mut operations = Vec::new();

        for (local, scalar) in self.reads_scalar.drain(..) {
            operations.push(Instruction::new(Operation::Copy(scalar), local));
            variables.push(local);
        }

        for op in self.operations.drain(..) {
            operations.push(op);
        }

        ScopeProcessing {
            variables,
            operations,
        }
        .optimize()
    }

    pub fn new_local_index(&self) -> u32 {
        self.allocator.new_local_index()
    }

    fn new_shared_index(&self) -> Id {
        self.shared_memories.len() as Id
    }

    fn new_const_array_index(&self) -> Id {
        self.const_arrays.len() as Id
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
    pub fn create_shared<I: Into<Item>>(&mut self, item: I, shared_memory_size: u32) -> Variable {
        let item = item.into();
        let index = self.new_shared_index();
        let shared_memory = Variable::new(
            VariableKind::SharedMemory {
                id: index,
                length: shared_memory_size,
            },
            item,
        );
        self.shared_memories.push(shared_memory);
        shared_memory
    }

    /// Create a shared variable of the given [item type](Item).
    pub fn create_const_array<I: Into<Item>>(&mut self, item: I, data: Vec<Variable>) -> Variable {
        let item = item.into();
        let index = self.new_const_array_index();
        let const_array = Variable::new(
            VariableKind::ConstantArray {
                id: index,
                length: data.len() as u32,
            },
            item,
        );
        self.const_arrays.push((const_array, data));
        const_array
    }

    /// Create a local array of the given [item type](Item).
    pub fn create_local_array<I: Into<Item>>(&mut self, item: I, array_size: u32) -> Variable {
        let local_array = self.allocator.create_local_array(item.into(), array_size);
        self.add_local_array(*local_array);
        *local_array
    }

    pub fn add_local_array(&mut self, var: Variable) {
        self.local_arrays.push(var);
    }
}
