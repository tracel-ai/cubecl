use alloc::{rc::Rc, string::String, vec::Vec};
use core::{
    any::{TypeId, type_name},
    cell::UnsafeCell,
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
use spin::LazyLock;
use std::{boxed::Box, vec};

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

/// SAFETY: This should be fine for parsing the AST, hopefully. There's just no good way to
/// have both owned and borrowed contexts in scopes.
#[derive(Clone)]
enum CtxHandle {
    Rc(Rc<UnsafeCell<Context>>),
    Ref(*mut Context),
}

impl CtxHandle {
    pub fn borrow(&self) -> &Context {
        match self {
            CtxHandle::Rc(cell) => unsafe { &*cell.get() },
            CtxHandle::Ref(ptr) => unsafe { &**ptr },
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn borrow_mut(&self) -> &mut Context {
        match self {
            CtxHandle::Rc(cell) => unsafe { &mut *cell.get() },
            CtxHandle::Ref(ptr) => unsafe { &mut **ptr },
        }
    }
}

/// SAFETY: This should be fine for parsing the AST, hopefully. There's just no good way to
/// have both owned and borrowed inserters in scopes.
enum InserterHandle {
    Owned(Box<UnsafeCell<dyn Inserter>>),
    Ref(*mut dyn Inserter),
}

impl InserterHandle {
    pub fn owned(inserter: impl Inserter + 'static) -> Self {
        Self::Owned(Box::new(UnsafeCell::new(inserter)))
    }

    #[allow(clippy::mut_from_ref)]
    pub fn borrow_mut(&self) -> &mut dyn Inserter {
        match self {
            InserterHandle::Owned(cell) => unsafe { &mut *cell.get() },
            InserterHandle::Ref(ptr) => unsafe { &mut **ptr },
        }
    }
}

/// The scope represents the region currently being parsed, as well as the current insertion point.
/// It is created from scratch for the initial codegen phase, or rebuilt from an existing scope and
/// a rewriter for passes/conversions.
///
/// All state apart from the insertion point is shared. Global state like the type map and errors
/// are stored in the context's auxiliary storage, so they can be reconstructed from the rewriter
/// state.
#[allow(missing_docs)]
pub struct Scope {
    ctx: CtxHandle,
    inserter: InserterHandle,
}

impl Debug for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scope").finish()
    }
}

pub fn ident(name: impl Into<String>) -> Identifier {
    Identifier::try_new(name.into()).unwrap()
}

static GLOBAL_STATE_IDENT: LazyLock<Identifier> = LazyLock::new(|| {
    let type_name = type_name::<GlobalState>();
    let sanitized = type_name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
    Identifier::try_new(sanitized).expect("Global state ident should be valid")
});

pub struct GlobalState {
    pub reference_arena: DropBump,
    pub errors: Vec<String>,

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

pub trait ContextExt {
    fn global_state(&self) -> &GlobalState;
    fn global_state_mut(&mut self) -> &mut GlobalState;
}

impl ContextExt for Context {
    fn global_state(&self) -> &GlobalState {
        let key = self.aux_data_map[&*GLOBAL_STATE_IDENT];
        self.aux_data[key].downcast_ref().unwrap()
    }

    fn global_state_mut(&mut self) -> &mut GlobalState {
        let key = self.aux_data_map[&*GLOBAL_STATE_IDENT];
        self.aux_data[key].downcast_mut().unwrap()
    }
}

fn new_context() -> Rc<UnsafeCell<Context>> {
    let mut context = Context::default();

    let module = ModuleOp::new(&mut context, ident("kernel"));
    let module_block = module.get_body(&context, 0);
    let mut module_inserter = OpInserter::new_at_block_end(module_block);

    let entry_func_ty = FunctionType::get(&mut context, vec![], vec![]);
    let entry_func = FuncOp::new(&mut context, ident("main"), entry_func_ty);
    module_inserter.append_op(&context, &entry_func);

    let state = GlobalState {
        reference_arena: Default::default(),
        module,
        module_inserter,
        entry_func,
        functions: Default::default(),
        typemap: Default::default(),
        sizemap: Default::default(),
        modes: Default::default(),
        target_properties: Default::default(),
        device_properties: Default::default(),
        errors: Default::default(),
    };
    let key = context.aux_data.insert(Box::new(state));
    context.aux_data_map.insert(GLOBAL_STATE_IDENT.clone(), key);

    Rc::new(UnsafeCell::new(context))
}

impl Debug for GlobalState {
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

/// Modes set and reset during expansion
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, TypeHash)]
pub struct InstructionModes {
    pub fp_math_mode: EnumSet<FastMath>,
}

impl Scope {
    /// Set the device properties.
    pub fn device_properties(&self, properties: &DeviceProperties) {
        self.state_mut().device_properties = Some(Rc::new(properties.clone()));
    }

    pub fn state(&self) -> &GlobalState {
        self.ctx().global_state()
    }

    pub fn state_mut(&self) -> &mut GlobalState {
        self.ctx_mut().global_state_mut()
    }

    pub fn ctx(&self) -> &Context {
        self.ctx.borrow()
    }

    // #[allow(clippy::mut_from_ref)]
    pub fn ctx_mut(&self) -> &mut Context {
        self.ctx.borrow_mut()
    }

    pub fn inserter(&self) -> &mut dyn Inserter {
        self.inserter.borrow_mut()
    }

    /// Create a parse scope that is at the root of a kernel definition.
    ///
    /// A local scope can be created with the [child](Self::child) method.
    pub fn root(_debug_enabled: bool) -> Self {
        let ctx = new_context();
        let inserter = {
            let ctx = unsafe { &*ctx.get() };
            let state = ctx.global_state();
            let entry_block = state.entry_func.get_entry_block(ctx);
            OpInserter::new_at_block_end(entry_block)
        };
        Self {
            ctx: CtxHandle::Rc(ctx),
            inserter: InserterHandle::owned(inserter),
        }
    }

    /// Create a rewrite scope from an existing context and inserter/rewriter
    pub fn from_context_and_inserter(
        ctx: &mut Context,
        inserter: &mut (impl Inserter + 'static),
    ) -> Self {
        let inserter: *mut dyn Inserter = inserter;
        Self {
            ctx: CtxHandle::Ref(ctx),
            inserter: InserterHandle::Ref(inserter),
        }
    }

    /// Create a new mutable local variable of type specified by `value_ty`.
    pub fn create_local_mut(&self, value_ty: impl Into<Ptr<TypeObj>>) -> Value {
        let value_ty = value_ty.into();
        let ctx = self.ctx_mut();
        let align = value_ty.align(ctx);
        let op = DeclareVariableOp::new(
            ctx,
            TypeAttr::new(value_ty),
            AddressSpace::Local.into(),
            align.into(),
        );
        let out = op.get_result(ctx);
        self.inserter().append_op(ctx, &op);
        out
    }

    /// Create a shared variable of the given item type.
    pub fn create_shared(
        &self,
        value_ty: impl Into<Ptr<TypeObj>>,
        alignment: Option<usize>,
    ) -> Value {
        let value_ty = value_ty.into();
        let ctx = self.ctx_mut();
        let align = alignment.unwrap_or_else(|| value_ty.align(ctx));
        let op = DeclareVariableOp::new(
            ctx,
            TypeAttr::new(value_ty),
            AddressSpace::Local.into(),
            align.into(),
        );
        let out = op.get_result(ctx);
        self.inserter().append_op(ctx, &op);
        out
    }

    /// Create a new function.
    pub fn create_function(&self, func: FuncOp) -> usize {
        // We know state doesn't overlap with the stuff that's used in `append_op` so this is safe
        let ctx = self.ctx();
        let state = self.ctx_mut().global_state_mut();
        let func_id = state.functions.len();
        state.module_inserter.append_op(ctx, &func);
        state.functions.push(func);
        func_id
    }

    /// Register an [`Instruction`] into the scope.
    pub fn register(&self, op: &dyn Op) {
        let ctx = self.ctx();
        self.inserter().append_op(ctx, op);
    }

    /// Add a value to the global arena so we can create a kernel-wide reference to it.
    /// The reference is the same as the type for simplicity, but is only valid for the duration of
    /// the root scope. Ensure the reference lifetime is shortened to the lifetime of the underlying
    /// variable being referenced.
    pub fn create_kernel_ref<'a, T>(&self, value: T) -> &'a mut T
    where
        T: 'a,
    {
        let state = self.state_mut();
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
        let state = self.state_mut();

        state.typemap.insert(TypeId::of::<T>(), elem);
    }

    /// Register the comptime size for the given generic size.
    pub fn register_size<T: 'static>(&self, size: usize) {
        let state = self.state_mut();

        state.sizemap.insert(TypeId::of::<T>(), size);
    }

    /// Register the type and size of a scalarizable type
    pub fn register_value_type<T: 'static, N: 'static>(&self, value: Value) {
        let ty = value.get_type(self.ctx());
        let scalar_ty = ty.scalar_ty(self.ctx());
        let vector_size = ty.vector_size(self.ctx());
        let storage_ty = {
            let ctx = self.ctx();
            let scalar_ty = scalar_ty.deref(ctx);
            let scalar = type_cast::<dyn ScalarType>(scalar_ty.as_ref()).unwrap();
            scalar.storage_type(ctx)
        };
        self.register_type::<T>(storage_ty);
        self.register_size::<N>(vector_size);
    }

    /// Create an empty child scope.
    pub fn child(&self, inserter: impl Inserter + 'static) -> Self {
        Self {
            ctx: self.ctx.clone(),
            inserter: InserterHandle::owned(inserter),
        }
    }

    // Adds a validation error.
    pub fn push_error(&self, msg: impl Into<String>) {
        self.state_mut().errors.push(msg.into());
    }

    /// Returns all validation errors.
    pub fn pop_errors(&self) -> Vec<String> {
        core::mem::take(&mut self.state_mut().errors)
    }

    /// Obtain the index-th buffer
    pub fn global(&self, id: usize, value_ty: Ptr<TypeObj>) -> Value {
        let ctx = self.ctx_mut();
        let state = self.state_mut();

        let ty_arr = RuntimeArrayType::get(ctx, value_ty);
        let ty = PointerType::get(ctx, ty_arr.into(), AddressSpace::Global(id));

        let mut arg_types = {
            let current_func_ty = state.entry_func.get_type(ctx).deref(ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            current_func_ty.arg_types()
        };

        arg_types.insert(id, ty.into());
        let new_func_ty = FunctionType::get(ctx, arg_types, vec![]);
        state
            .entry_func
            .set_attr_func_type(ctx, TypeAttr::new(new_func_ty.into()));

        let entry_block = state.entry_func.get_entry_block(ctx).deref(ctx);
        entry_block.get_argument(id)
    }

    /// Obtain the index-th tensor map
    pub fn tensor_map(&self, id: usize) -> Value {
        let ctx = self.ctx_mut();
        let state = self.state_mut();
        let ty = TensorMapType::get(ctx);
        let mut arg_types = {
            let current_func_ty = state.entry_func.get_type(ctx).deref(ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            current_func_ty.arg_types()
        };

        arg_types.insert(id, ty.into());
        let new_func_ty = FunctionType::get(ctx, arg_types, vec![]);
        state
            .entry_func
            .set_attr_func_type(ctx, TypeAttr::new(new_func_ty.into()));
        let entry_block = state.entry_func.get_entry_block(ctx).deref(ctx);
        entry_block.get_argument(id)
    }

    pub fn extract_field(&self, aggregate: Value, field: usize) -> Value {
        let ctx = self.ctx_mut();
        let op = AggregateExtractOp::new(ctx, aggregate, field.into());
        self.register(&op);
        op.get_result(ctx)
    }

    pub fn const_usize(&self, value: usize) -> Value {
        let op = ConstantOp::new(self.ctx_mut(), IndexAttr::new(value).into());
        self.register(&op);
        op.get_result(self.ctx())
    }
}

impl Display for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let ctx = self.ctx();
        let state = ctx.global_state();
        write!(f, "{}", state.module.disp(ctx))
    }
}
