use alloc::collections::BTreeMap;
use alloc::{borrow::Cow, rc::Rc, string::String, string::ToString, vec::Vec};
use bumpalo::Bump;
use core::{
    any::TypeId,
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
};
use derive_more::{Eq, PartialEq};
use enumset::EnumSet;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;

use crate::{
    AggregateExtractOperands, AggregateKind, BarrierLevel, CubeFnSource, DeviceProperties,
    FastMath, Function, Matrix, Operation, OperationReflect, Processor, SemanticType, SourceLoc,
    StorageType, TargetProperties, TypeHash,
};

use super::{
    Allocator, Id, Instruction, Type, Variable, VariableKind, processing::ScopeProcessing,
};

pub type TypeMap = HashMap<TypeId, StorageType>;
pub type SizeMap = HashMap<TypeId, usize>;

/// The scope is the main [`crate::Operation`] and [`crate::Variable`] container that simplify
/// the process of reading inputs, creating local variables and adding new operations.
///
/// Notes:
///
/// This type isn't responsible for creating shader bindings and figuring out which
/// variable can be written to.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
#[allow(missing_docs)]
pub struct Scope {
    validation_errors: ValidationErrors,
    pub depth: u8,
    pub instructions: RefCell<Vec<Instruction>>,
    pub return_value: Option<Variable>,
    pub locals: RefCell<Vec<Variable>>,
    pub const_arrays: RefCell<Vec<(Variable, Vec<Variable>)>>,
    pub debug: DebugInfo,

    #[cfg_attr(feature = "serde", serde(skip))]
    pub global_state: GlobalState,
}

pub type GlobalState = Rc<RefCell<GlobalStateInner>>;

#[derive(Debug, PartialEq, Eq, TypeHash, Default)]
pub struct GlobalStateInner {
    #[partial_eq(skip)]
    #[eq(skip)]
    pub reference_arena: Bump,
    pub allocator: Allocator,

    pub functions: BTreeMap<Id, Function>,
    pub typemap: TypeMap,
    pub sizemap: SizeMap,
    pub modes: InstructionModes,
    pub target_properties: TargetProperties,
    pub device_properties: Option<Rc<DeviceProperties>>,
}

impl GlobalStateInner {
    pub fn clone_deep(&self) -> Self {
        Self {
            reference_arena: Bump::new(),
            allocator: self.allocator.clone_deep(),
            functions: self.functions.clone(),
            typemap: self.typemap.clone(),
            sizemap: self.sizemap.clone(),
            modes: self.modes,
            target_properties: self.target_properties.clone(),
            device_properties: self.device_properties.clone(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct ValidationErrors {
    errors: Rc<RefCell<Vec<String>>>,
}

/// Debug related fields, most of these are global
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct DebugInfo {
    pub enabled: bool,
    pub sources: Rc<RefCell<HashSet<CubeFnSource>>>,
    pub variable_names: Rc<RefCell<HashMap<Variable, Cow<'static, str>>>>,
    pub source_loc: RefCell<Option<SourceLoc>>,
    pub entry_loc: RefCell<Option<SourceLoc>>,
}

/// Modes set and reset during expansion
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, TypeHash)]
pub struct InstructionModes {
    pub fp_math_mode: EnumSet<FastMath>,
}

impl core::hash::Hash for Scope {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        self.depth.hash(ra_expand_state);
        self.instructions.borrow().hash(ra_expand_state);
        self.locals.borrow().hash(ra_expand_state);
        self.const_arrays.borrow().hash(ra_expand_state);
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
    /// Set the device properties.
    pub fn device_properties(&self, properties: &DeviceProperties) {
        self.state_mut().device_properties = Some(Rc::new(properties.clone()));
    }

    pub fn state(&self) -> Ref<'_, GlobalStateInner> {
        self.global_state.borrow()
    }

    pub fn state_mut(&self) -> RefMut<'_, GlobalStateInner> {
        self.global_state.borrow_mut()
    }

    /// Create a scope that is at the root of a kernel definition.
    ///
    /// A local scope can be created with the [child](Self::child) method.
    pub fn root(debug_enabled: bool) -> Self {
        Self {
            validation_errors: ValidationErrors {
                errors: Rc::new(RefCell::new(Vec::new())),
            },
            depth: 0,
            instructions: Default::default(),
            return_value: None,
            locals: Default::default(),
            const_arrays: Default::default(),
            debug: DebugInfo {
                enabled: debug_enabled,
                sources: Default::default(),
                variable_names: Default::default(),
                source_loc: Default::default(),
                entry_loc: Default::default(),
            },
            global_state: Default::default(),
        }
    }

    /// Use existing state.
    pub fn with_global_state(mut self, global_state: GlobalState) -> Self {
        self.global_state = global_state;
        self
    }

    /// Create a new matrix element.
    pub fn create_matrix(&self, matrix: Matrix) -> Variable {
        let matrix = self.state().allocator.create_matrix(matrix);
        self.add_matrix(matrix);
        matrix
    }

    pub fn add_matrix(&self, variable: Variable) {
        self.locals.borrow_mut().push(variable);
    }

    /// Create a new pipeline element.
    pub fn create_pipeline(&self, num_stages: u8) -> Variable {
        self.state().allocator.create_pipeline(num_stages)
    }

    /// Create a new barrier element.
    pub fn create_barrier_token(&self, id: Id, level: BarrierLevel) -> Variable {
        Variable::new(
            VariableKind::BarrierToken { id, level },
            Type::semantic(SemanticType::BarrierToken),
        )
    }

    /// Create a mutable variable of the given item type.
    pub fn create_local_mut<I: Into<Type>>(&self, item: I) -> Variable {
        self.state().allocator.create_local_mut(item.into())
    }

    /// Create a new restricted variable. The variable is
    /// Useful for _for loops_ and other algorithms that require the control over initialization.
    pub fn create_local_restricted(&self, ty: Type) -> Variable {
        self.state().allocator.create_local_restricted(ty)
    }

    /// Create a new immutable variable.
    pub fn create_local(&self, ty: Type) -> Variable {
        self.state().allocator.create_local(ty)
    }

    /// Create a new immutable variable of aggregate type.
    pub fn create_aggregate(&self, ty: Type, kind: AggregateKind) -> Variable {
        let id = self.state().allocator.new_local_index();
        Variable::new(
            VariableKind::Aggregate {
                id,
                aggregate_kind: kind,
            },
            ty,
        )
    }

    /// Create a new function.
    pub fn create_function(&self, explicit_params: Vec<Variable>, scope: Scope) -> Id {
        let id = self.state().allocator.new_local_index();
        self.state_mut().functions.insert(
            id,
            Function {
                explicit_params,
                scope,
            },
        );
        id
    }

    /// Register an [`Instruction`] into the scope.
    pub fn register<T: Into<Instruction>>(&self, instruction: T) {
        let mut inst = instruction.into();
        inst.operation.sanitize_args(self);
        inst.source_loc = self.debug.source_loc.borrow().clone();
        inst.modes = self.state().modes;
        self.instructions.borrow_mut().push(inst)
    }

    /// Add a value to the global arena so we can create a kernel-wide reference to it.
    /// The reference is `'static` for simplicity, but is only valid for the duration of the root scope.
    /// Ensure the reference lifetime is shortened to the lifetime of the underlying variable being
    /// referenced.
    pub fn create_kernel_ref<T: 'static>(&self, var: T) -> &'static mut T {
        let state = self.global_state.borrow();
        let reference = state.reference_arena.alloc(var);
        unsafe { core::mem::transmute(reference) }
    }

    /// Resolve the element type of the given generic type.
    pub fn resolve_type<T: 'static>(&self) -> Option<StorageType> {
        let state = self.state();
        let result = state.typemap.get(&TypeId::of::<T>());

        result.cloned()
    }

    /// Resolve the comptime size of the given generic size.
    pub fn resolve_size<T: 'static>(&self) -> Option<usize> {
        let state = self.state();
        let result = state.sizemap.get(&TypeId::of::<T>());

        result.cloned()
    }

    /// Register the element type for the given generic type.
    pub fn register_type<T: 'static>(&self, elem: StorageType) {
        let mut state = self.state_mut();

        state.typemap.insert(TypeId::of::<T>(), elem);
    }

    /// Register the comptime size for the given generic size.
    pub fn register_size<T: 'static>(&self, size: usize) {
        let mut state = self.state_mut();

        state.sizemap.insert(TypeId::of::<T>(), size);
    }

    /// Create an empty child scope.
    pub fn child(&self) -> Self {
        Self {
            validation_errors: self.validation_errors.clone(),
            depth: self.depth + 1,
            instructions: Default::default(),
            return_value: None,
            locals: Default::default(),
            const_arrays: Default::default(),
            debug: self.debug.clone(),
            global_state: self.global_state.clone(),
        }
    }

    // Adds a validation error.
    pub fn push_error(&self, msg: impl Into<String>) {
        self.validation_errors.errors.borrow_mut().push(msg.into());
    }

    /// Returns all validation errors.
    pub fn pop_errors(&self) -> Vec<String> {
        self.validation_errors.errors.replace_with(|_| Vec::new())
    }

    /// Returns the variables and operations to be declared and executed.
    ///
    /// Notes:
    ///
    /// New operations and variables can be created within the same scope without having name
    /// conflicts.
    pub fn process<'a>(
        &self,
        processors: impl IntoIterator<Item = &'a dyn Processor>,
    ) -> ScopeProcessing {
        self.global_state.borrow_mut().reference_arena.reset();
        let mut variables = core::mem::take(&mut *self.locals.borrow_mut());

        let mut instructions = Vec::new();

        for inst in self.instructions.borrow_mut().drain(..) {
            instructions.push(inst);
        }

        variables.extend(self.state().allocator.take_variables());

        let mut processing = ScopeProcessing {
            variables,
            instructions,
            global_state: self.global_state.clone(),
        };

        for p in processors {
            processing = p.transform(processing);
        }

        // Add variables added from processors
        processing
            .variables
            .extend(self.state().allocator.take_variables());

        processing
    }

    pub fn new_local_index(&self) -> u32 {
        self.state().allocator.new_local_index()
    }

    /// Create a shared variable of the given item type.
    pub fn create_shared<I: Into<Type>>(&self, item: I, alignment: Option<usize>) -> Variable {
        let item = item.into();
        let index = self.new_local_index();
        Variable::new(
            VariableKind::Shared {
                id: index,
                alignment,
            },
            item,
        )
    }

    /// Create a shared variable of the given item type.
    pub fn create_const_array<I: Into<Type>>(&self, item: I, data: Vec<Variable>) -> Variable {
        let item = item.into();
        let index = self.new_local_index();
        let const_array = Variable::new(
            VariableKind::ConstantArray {
                id: index,
                length: data.len(),
                unroll_factor: 1,
            },
            item,
        );
        self.const_arrays.borrow_mut().push((const_array, data));
        const_array
    }

    /// Obtain the index-th buffer
    pub fn global(&self, id: Id, item: Type) -> Variable {
        Variable::new(
            VariableKind::GlobalBuffer(id),
            Type::DynamicArray(item.intern(), crate::AddressSpace::Global(id)),
        )
    }

    /// Obtain the index-th scalar
    pub fn scalar(&self, id: Id, storage: StorageType) -> Variable {
        Variable::new(VariableKind::GlobalScalar(id), Type::new(storage))
    }

    pub fn update_source(&self, source: CubeFnSource) {
        if self.debug.enabled {
            self.debug.sources.borrow_mut().insert(source.clone());
            *self.debug.source_loc.borrow_mut() = Some(SourceLoc {
                line: source.line,
                column: source.column,
                source,
            });
            if self.debug.entry_loc.borrow().is_none() {
                *self.debug.entry_loc.borrow_mut() = self.debug.source_loc.borrow().clone();
            }
        }
    }

    pub fn register_all(&self, instructions: impl IntoIterator<Item = Instruction>) {
        self.instructions.borrow_mut().extend(instructions);
    }

    pub fn take_instructions(&self) -> Vec<Instruction> {
        core::mem::take(&mut *self.instructions.borrow_mut())
    }

    pub fn update_span(&self, line: u32, col: u32) {
        if let Some(loc) = self.debug.source_loc.borrow_mut().as_mut() {
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

    pub fn extract_field(&self, aggregate: Variable, ty: Type, field: usize) -> Variable {
        if !matches!(aggregate.kind, VariableKind::Aggregate { .. }) {
            panic!(
                "Tried extracting field from non-aggregate {aggregate}.\nCurrent state:\n{}",
                self.instructions.borrow().iter().join("\n")
            )
        }
        let out = self.create_local(ty);
        self.register(Instruction::new(
            Operation::ExtractAggregateField(AggregateExtractOperands { aggregate, field }),
            out,
        ));
        out
    }
}

impl Display for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "{{")?;
        for instruction in self.instructions.borrow().iter() {
            let instruction_str = instruction.to_string();
            if !instruction_str.is_empty() {
                writeln!(
                    f,
                    "{}{}",
                    "    ".repeat(self.depth as usize + 1),
                    instruction_str,
                )?;
            }
        }
        write!(f, "{}}}", "    ".repeat(self.depth as usize))?;
        Ok(())
    }
}
