use alloc::{rc::Rc, string::String, vec::Vec};
use core::{
    any::TypeId,
    cell::{Ref, RefCell, RefMut},
    fmt::{Debug, Display},
};
use derive_more::{Eq, PartialEq};
use enumset::EnumSet;
use hashbrown::HashMap;
use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
        ops::{ConstantOp, FuncOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        inserter::{IRInserter, Inserter},
        listener::DummyListener,
    },
    op::Op,
    printable::Printable,
    r#type::{TypeObj, Typed, type_cast},
    value::Value,
};
use std::vec;

use crate::{
    AddressSpace, DeviceProperties, FastMath, StorageType, TargetProperties, TypeHash,
    arena::DropBump,
    attributes::IndexAttr,
    dialect::{general::AggregateExtractOp, memory::DeclareVariableOp},
    interfaces::{ScalarType, TypedExt},
    types::{PointerType, RuntimeArrayType, cuda::TensorMapType},
};

pub type TypeMap = HashMap<TypeId, StorageType>;
pub type SizeMap = HashMap<TypeId, usize>;

pub type OpInserter = IRInserter<DummyListener>;

/// The scope is the main [`crate::Operation`] and [`crate::Value`] container that simplify
/// the process of reading inputs, creating local variables and adding new operations.
///
/// Notes:
///
/// This type isn't responsible for creating shader bindings and figuring out which
/// variable can be written to.
#[allow(missing_docs)]
pub struct Scope {
    validation_errors: ValidationErrors,
    pub inserter: RefCell<OpInserter>,
    pub global_state: GlobalState,
}

impl Debug for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scope")
            .field("validation_errors", &self.validation_errors)
            .field("global_state", &self.global_state)
            .finish()
    }
}

pub type GlobalState = Rc<RefCell<GlobalStateInner>>;

pub fn ident(name: impl Into<String>) -> Identifier {
    Identifier::try_new(name.into()).unwrap()
}

pub struct GlobalStateInner {
    pub reference_arena: DropBump,

    pub ctx: Context,
    pub module: ModuleOp,
    pub module_inserter: OpInserter,
    pub entry_func: FuncOp,
    pub functions: Vec<FuncOp>,
    pub typemap: TypeMap,
    pub sizemap: SizeMap,
    pub modes: InstructionModes,
    pub target_properties: TargetProperties,
    pub device_properties: Option<Rc<DeviceProperties>>,
}

impl GlobalStateInner {
    pub fn new() -> GlobalState {
        let mut context = Context::default();

        let module = ModuleOp::new(&mut context, ident("kernel"));
        let module_block = module.get_body(&context, 0);
        let mut module_inserter = OpInserter::new_at_block_end(module_block);

        let entry_func_ty = FunctionType::get(&mut context, vec![], vec![]);
        let entry_func = FuncOp::new(&mut context, ident("main"), entry_func_ty);
        module_inserter.append_op(&context, &entry_func);

        let inner = Self {
            reference_arena: Default::default(),
            ctx: context,
            module,
            module_inserter,
            entry_func,
            functions: Default::default(),
            typemap: Default::default(),
            sizemap: Default::default(),
            modes: Default::default(),
            target_properties: Default::default(),
            device_properties: Default::default(),
        };
        Rc::new(RefCell::new(inner))
    }
}

impl Debug for GlobalStateInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlobalStateInner")
            .field("reference_arena", &self.reference_arena)
            .field("typemap", &self.typemap)
            .field("sizemap", &self.sizemap)
            .field("modes", &self.modes)
            .field("target_properties", &self.target_properties)
            .field("device_properties", &self.device_properties)
            .finish()
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct ValidationErrors {
    errors: Rc<RefCell<Vec<String>>>,
}

/// Modes set and reset during expansion
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, TypeHash)]
pub struct InstructionModes {
    pub fp_math_mode: EnumSet<FastMath>,
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

    pub fn ctx(&self) -> Ref<'_, Context> {
        Ref::map(self.state(), |state| &state.ctx)
    }

    pub fn ctx_mut(&self) -> RefMut<'_, Context> {
        RefMut::map(self.state_mut(), |state| &mut state.ctx)
    }

    /// Create a scope that is at the root of a kernel definition.
    ///
    /// A local scope can be created with the [child](Self::child) method.
    pub fn root(_debug_enabled: bool) -> Self {
        let global_state = GlobalStateInner::new();
        let inserter = {
            let state = global_state.borrow();
            let entry_block = state.entry_func.get_entry_block(&state.ctx);
            OpInserter::new_at_block_end(entry_block)
        };
        Self {
            validation_errors: ValidationErrors {
                errors: Rc::new(RefCell::new(Vec::new())),
            },
            inserter: RefCell::new(inserter),
            global_state: GlobalStateInner::new(),
        }
    }

    /// Use existing state.
    pub fn with_global_state(mut self, global_state: GlobalState) -> Self {
        self.global_state = global_state;
        self
    }

    /// Create a new mutable local variable of type specified by `value_ty`.
    pub fn create_local_mut(&self, value_ty: impl Into<Ptr<TypeObj>>) -> Value {
        let value_ty = value_ty.into();
        let mut ctx = self.ctx_mut();
        let align = value_ty.align(&ctx);
        let op = DeclareVariableOp::new(
            &mut ctx,
            TypeAttr::new(value_ty),
            AddressSpace::Local.into(),
            align.into(),
        );
        let out = op.get_result(&ctx);
        self.inserter.borrow_mut().append_op(&ctx, &op);
        out
    }

    /// Create a shared variable of the given item type.
    pub fn create_shared(
        &self,
        value_ty: impl Into<Ptr<TypeObj>>,
        alignment: Option<usize>,
    ) -> Value {
        let value_ty = value_ty.into();
        let mut ctx = self.ctx_mut();
        let align = alignment.unwrap_or_else(|| value_ty.align(&ctx));
        let op = DeclareVariableOp::new(
            &mut ctx,
            TypeAttr::new(value_ty),
            AddressSpace::Local.into(),
            align.into(),
        );
        let out = op.get_result(&ctx);
        self.inserter.borrow_mut().append_op(&ctx, &op);
        out
    }

    /// Create a new function.
    pub fn create_function(&self, func: FuncOp) -> usize {
        let mut state = self.state_mut();
        let state = &mut *state;
        let func_id = state.functions.len();
        state.module_inserter.append_op(&state.ctx, &func);
        state.functions.push(func);
        func_id
    }

    /// Register an [`Instruction`] into the scope.
    pub fn register(&self, op: &dyn Op) {
        let ctx = self.ctx();
        self.inserter.borrow_mut().append_op(&ctx, op);
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

    /// Register the type and size of a scalarizable type
    pub fn register_value_type<T: 'static, N: 'static>(&self, value: Value) {
        let ty = value.get_type(&self.ctx());
        let scalar_ty = ty.scalar_ty(&self.ctx());
        let vector_size = ty.vector_size(&self.ctx());
        let storage_ty = {
            let ctx = self.ctx();
            let scalar_ty = scalar_ty.deref(&ctx);
            let scalar = type_cast::<dyn ScalarType>(scalar_ty.as_ref()).unwrap();
            scalar.storage_type(&ctx)
        };
        self.register_type::<T>(storage_ty);
        self.register_size::<N>(vector_size);
    }

    /// Create an empty child scope.
    pub fn child(&self, inserter: OpInserter) -> Self {
        Self {
            validation_errors: self.validation_errors.clone(),
            inserter: RefCell::new(inserter),
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

    /// Obtain the index-th buffer
    pub fn global(&self, id: usize, value_ty: Ptr<TypeObj>) -> Value {
        let mut state = self.state_mut();
        let state = &mut *state;

        let ty_arr = RuntimeArrayType::get(&mut state.ctx, value_ty);
        let ty = PointerType::get(&mut state.ctx, ty_arr.into(), AddressSpace::Global(id));

        let mut arg_types = {
            let current_func_ty = state.entry_func.get_type(&state.ctx).deref(&state.ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            current_func_ty.arg_types()
        };

        arg_types.insert(id, ty.into());
        let new_func_ty = FunctionType::get(&mut state.ctx, arg_types, vec![]);
        state
            .entry_func
            .set_attr_func_type(&state.ctx, TypeAttr::new(new_func_ty.into()));

        let entry_block = state
            .entry_func
            .get_entry_block(&state.ctx)
            .deref(&state.ctx);
        entry_block.get_argument(id)
    }

    /// Obtain the index-th tensor map
    pub fn tensor_map(&self, id: usize) -> Value {
        let mut state = self.state_mut();
        let ty = TensorMapType::get(&state.ctx);
        let mut arg_types = {
            let current_func_ty = state.entry_func.get_type(&state.ctx).deref(&state.ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            current_func_ty.arg_types()
        };

        arg_types.insert(id, ty.into());
        let new_func_ty = FunctionType::get(&mut state.ctx, arg_types, vec![]);
        state
            .entry_func
            .set_attr_func_type(&state.ctx, TypeAttr::new(new_func_ty.into()));
        let entry_block = state
            .entry_func
            .get_entry_block(&state.ctx)
            .deref(&state.ctx);
        entry_block.get_argument(id)
    }

    pub fn extract_field(&self, aggregate: Value, field: usize) -> Value {
        let mut ctx = self.ctx_mut();
        let op = AggregateExtractOp::new(&mut ctx, aggregate, field.into());
        self.register(&op);
        op.get_result(&ctx)
    }

    pub fn const_usize(&self, value: usize) -> Value {
        let op = ConstantOp::new(&mut self.ctx_mut(), IndexAttr::new(value).into());
        self.register(&op);
        op.get_result(&self.ctx())
    }
}

impl Display for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let state = self.state();
        write!(f, "{}", state.module.disp(&state.ctx))
    }
}
