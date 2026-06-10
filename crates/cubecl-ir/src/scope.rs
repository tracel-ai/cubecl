use alloc::collections::BTreeMap;
use alloc::{borrow::Cow, rc::Rc, string::String, string::ToString, vec::Vec};
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
    AddressSpace, AggregateExtractOperands, CubeFnSource, DeviceProperties, FastMath, Function,
    OpaqueType, Operation, OperationReflect, Processor, SourceLoc, StorageType, TargetProperties,
    TypeHash, arena::DropBump,
};

use super::{Allocator, Id, Instruction, Type, Value, processing::ScopeProcessing};

pub type TypeMap = HashMap<TypeId, StorageType>;
pub type SizeMap = HashMap<TypeId, usize>;

/// The scope is the main [`crate::Operation`] and [`crate::Value`] container that simplify
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
    pub return_value: Option<Value>,
    pub locals: RefCell<Vec<Value>>,
    pub debug: DebugInfo,

    #[cfg_attr(feature = "serde", serde(skip))]
    pub global_state: GlobalState,
}

pub type GlobalState = Rc<RefCell<GlobalStateInner>>;

#[derive(Debug, PartialEq, Eq, TypeHash, Default)]
pub struct GlobalStateInner {
    #[partial_eq(skip)]
    #[eq(skip)]
    pub reference_arena: DropBump,
    pub allocator: Allocator,

    pub global_args: Vec<Value>,
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
            reference_arena: DropBump::new(),
            allocator: self.allocator.clone_deep(),
            global_args: self.global_args.clone(),
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
    pub value_names: Rc<RefCell<HashMap<Value, Cow<'static, str>>>>,
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
            debug: DebugInfo {
                enabled: debug_enabled,
                sources: Default::default(),
                value_names: Default::default(),
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

    /// Create a new immutable value of type specified by `ty`.
    pub fn create_value(&self, ty: Type) -> Value {
        let id = self.new_local_index();
        Value::new(id, ty)
    }

    /// Create a new mutable local variable of type specified by `value_ty`.
    pub fn create_local_mut(&self, value_ty: impl Into<Type>) -> Value {
        let value_ty = value_ty.into();
        let ty = Type::Pointer(value_ty.intern(), AddressSpace::Local);
        let out = self.create_value(ty);
        self.register(Instruction::new(
            Operation::DeclareVariable {
                value_ty,
                addr_space: AddressSpace::Local,
                alignment: value_ty.align(),
            },
            out,
        ));
        out
    }

    /// Create a shared variable of the given item type.
    pub fn create_shared(&self, value_ty: impl Into<Type>, alignment: Option<usize>) -> Value {
        let value_ty = value_ty.into();
        let ty = Type::Pointer(value_ty.intern(), AddressSpace::Shared);
        let out = self.create_value(ty);
        self.register(Instruction::new(
            Operation::DeclareVariable {
                value_ty,
                addr_space: AddressSpace::Shared,
                alignment: alignment.unwrap_or_else(|| value_ty.align()),
            },
            out,
        ));
        out
    }

    /// Create a new function.
    pub fn create_function(&self, explicit_params: Vec<Value>, scope: Scope) -> Id {
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
    /// The reference is the same as the type for simplicity, but is only valid for the duration of
    /// the root scope. Ensure the reference lifetime is shortened to the lifetime of the underlying
    /// variable being referenced.
    pub fn create_kernel_ref<'a, T>(&self, value: T) -> &'a mut T
    where
        T: 'a,
    {
        let mut state = self.state_mut();
        let reference = state.reference_arena.alloc(value);
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

    /// Returns the operations to be declared and executed.
    ///
    /// Notes:
    ///
    /// New operations can be created within the same scope without having name
    /// conflicts.
    pub fn process<'a>(
        &self,
        processors: impl IntoIterator<Item = &'a dyn Processor>,
    ) -> ScopeProcessing {
        self.global_state.borrow_mut().reference_arena.reset();

        let mut instructions = Vec::new();

        for inst in self.instructions.borrow_mut().drain(..) {
            instructions.push(inst);
        }

        let mut processing = ScopeProcessing {
            instructions,
            global_state: self.global_state.clone(),
        };

        for p in processors {
            processing = p.transform(processing);
        }

        processing
    }

    pub fn new_local_index(&self) -> u32 {
        self.state().allocator.new_local_index()
    }

    /// Obtain the index-th buffer
    pub fn global(&self, id: Id, value_ty: Type) -> Value {
        let ty_arr = Type::DynamicArray(value_ty.intern());
        let ty = Type::Pointer(ty_arr.intern(), AddressSpace::Global(id));
        let value = self.create_value(ty);
        self.state_mut().global_args.insert(id as usize, value);
        value
    }

    /// Obtain the index-th tensor map
    pub fn tensor_map(&self, id: Id) -> Value {
        let ty = Type::Opaque(OpaqueType::TensorMap);
        let value = self.create_value(ty);
        self.state_mut().global_args.insert(id as usize, value);
        value
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

    pub fn update_value_name(&self, value: Value, name: impl Into<Cow<'static, str>>) {
        if self.debug.enabled {
            self.debug
                .value_names
                .borrow_mut()
                .insert(value, name.into());
        }
    }

    pub fn extract_field(&self, aggregate: Value, ty: Type, field: usize) -> Value {
        if !matches!(aggregate.ty, Type::Aggregate(..)) {
            panic!(
                "Tried extracting field from non-aggregate {aggregate}.\nCurrent state:\n{}",
                self.instructions.borrow().iter().join("\n")
            )
        }
        let out = self.create_value(ty);
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
