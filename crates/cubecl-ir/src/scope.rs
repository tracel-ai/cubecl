use alloc::{boxed::Box, format, rc::Rc, string::String, vec, vec::Vec};
use core::{
    any::{TypeId, type_name},
    cell::UnsafeCell,
    fmt::{Debug, Display},
};
use cubecl_common::{format::type_name_sanitized, stub::Mutex};
use derive_more::{Eq, PartialEq};
use enumset::EnumSet;
use hashbrown::HashMap;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
        ops::{ConstantOp, FuncOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, UnitType},
    },
    context::{AuxDataIndex, Context},
    identifier::Identifier,
    irbuild::{
        inserter::{IRInserter, Inserter},
        listener::DummyListener,
    },
    op::Op,
    printable::Printable,
    r#type::{TypeHandle, Typed, type_cast},
    value::Value,
};
use spin::LazyLock;

use crate::{
    AddressSpace, DeviceProperties, FastMath, StorageType, TargetProperties, TypeHash,
    arena::DropBump,
    attributes::{
        ATTR_BUFFER_BINDING, BufferBindingAttr, EntrypointAbiAttr, EntrypointInterface,
        FuncInterface, IndexAttr,
    },
    dialect::{general::AggregateExtractOp, memory::DeclareVariableOp},
    interfaces::{ScalarType, TypedExt},
    settings::KernelSettings,
    types::{PointerType, RuntimeArrayType, cuda::TensorMapType},
};

pub type Types = HashMap<TypeId, StorageType>;
pub type Sizes = HashMap<TypeId, usize>;

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

pub struct GlobalState {
    pub reference_arena: DropBump,
    pub errors: Vec<String>,

    pub module: ModuleOp,
    pub module_inserter: OpInserter,
    pub entry_func: FuncOp,
    pub functions: Vec<FuncOp>,
    pub typemap: Types,
    pub sizemap: Sizes,
    pub modes: InstructionModes,
    pub target_properties: TargetProperties,
    pub device_properties: Option<Rc<DeviceProperties>>,
}

impl GlobalState {
    /// Register the element type for the given generic type.
    pub fn register_type<T: 'static>(&mut self, elem: StorageType) {
        self.typemap.insert(TypeId::of::<T>(), elem);
    }
}

static TY_IDENTS: LazyLock<Mutex<HashMap<TypeId, Identifier>>> = LazyLock::new(Default::default);

fn ty_ident<T: 'static>() -> Identifier {
    let mut idents = TY_IDENTS.lock().unwrap();
    let ident = idents
        .entry(TypeId::of::<T>())
        .or_insert_with(|| Identifier::try_from(type_name_sanitized::<T>()).unwrap());
    ident.clone()
}

fn ty_key<T: 'static>(ctx: &Context) -> Option<AuxDataIndex> {
    let mut idents = TY_IDENTS.lock().unwrap();
    let ident = idents
        .entry(TypeId::of::<T>())
        .or_insert_with(|| Identifier::try_from(type_name_sanitized::<T>()).unwrap());
    ctx.aux_data_map.get(ident).copied()
}

pub trait ContextExt {
    fn aux_ty<T: 'static>(&self) -> &T;
    fn aux_ty_mut<T: 'static>(&mut self) -> &mut T;
    fn set_aux_ty<T: 'static>(&mut self, value: T);
}

impl ContextExt for Context {
    fn aux_ty<T: 'static>(&self) -> &T {
        let key = ty_key::<T>(self)
            .ok_or_else(|| format!("Key for {} should exist", type_name::<T>()))
            .unwrap();
        self.aux_data[key].downcast_ref().unwrap()
    }

    fn aux_ty_mut<T: 'static>(&mut self) -> &mut T {
        let key = ty_key::<T>(self)
            .ok_or_else(|| format!("Key for {} should exist", type_name::<T>()))
            .unwrap();
        self.aux_data[key].downcast_mut().unwrap()
    }

    fn set_aux_ty<T: 'static>(&mut self, value: T) {
        if let Some(key) = ty_key::<T>(self) {
            *self.aux_data.get_mut(key).unwrap() = Box::new(value);
        } else {
            let ident = ty_ident::<T>();
            let key = self.aux_data.insert(Box::new(value));
            self.aux_data_map.insert(ident, key);
        }
    }
}

pub trait FuncOpExt {
    fn push_argument(&self, ctx: &Context, ty: TypeHandle) -> usize;
}

impl FuncOpExt for FuncOp {
    fn push_argument(&self, ctx: &Context, ty: TypeHandle) -> usize {
        let id = BasicBlock::push_argument(self.get_entry_block(ctx), ctx, ty);

        let (mut arg_types, res_types) = {
            let current_func_ty = self.get_type(ctx).deref(ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            (current_func_ty.arg_types(), current_func_ty.res_types())
        };

        arg_types.insert(id, ty);
        let new_func_ty = FunctionType::get(ctx, arg_types, res_types);
        self.set_attr_func_type(ctx, TypeAttr::new(new_func_ty.into()));
        id
    }
}

fn new_context(settings: KernelSettings) -> Rc<UnsafeCell<Context>> {
    let mut ctx = Context::default();

    let module = ModuleOp::new(&mut ctx, ident("kernel"));
    let module_block = module.get_body(&ctx, 0);
    let mut module_inserter = OpInserter::new_at_block_end(module_block);

    // Start out empty and fill in once args register themselves
    let entry_func_ty = FunctionType::get(&ctx, vec![], vec![UnitType::get(&ctx).into()]);
    let entry_name = ident(settings.kernel_name);
    let abi = EntrypointAbiAttr::new(
        settings.cube_dim,
        settings.cluster_dim,
        settings.address_type,
    );
    let entry_func = FuncOp::new(&mut ctx, entry_name, entry_func_ty);
    entry_func.set_entrypoint_abi(&mut ctx, abi);
    module_inserter.append_op(&ctx, &entry_func);

    let mut state = GlobalState {
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
    settings.address_type.register(&mut state);

    ctx.set_aux_ty(state);
    Rc::new(UnsafeCell::new(ctx))
}

/// Create a dummy context that can't be used for actual codegen. Useful for registering and
/// resolving types without making the interface for that overly complex.
/// Maybe we can replace the `&Scope` with a type registration trait on that function only at some point.
fn dummy_context() -> Rc<UnsafeCell<Context>> {
    let mut ctx = Context::default();

    let module = ModuleOp::new(&mut ctx, ident("dummy_module"));
    let module_block = module.get_body(&ctx, 0);
    let mut module_inserter = OpInserter::new_at_block_end(module_block);

    let entry_func_ty = FunctionType::get(&ctx, vec![], vec![UnitType::get(&ctx).into()]);
    let entry_name = ident("dummy_entry");
    let entry_func = FuncOp::new(&mut ctx, entry_name, entry_func_ty);
    module_inserter.append_op(&ctx, &entry_func);

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

    ctx.set_aux_ty(state);
    Rc::new(UnsafeCell::new(ctx))
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
        self.ctx().aux_ty()
    }

    pub fn state_mut(&self) -> &mut GlobalState {
        self.ctx_mut().aux_ty_mut()
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
    pub fn root(settings: KernelSettings) -> Self {
        let ctx = new_context(settings);
        let inserter = {
            let ctx = unsafe { &*ctx.get() };
            let state = ctx.aux_ty::<GlobalState>();
            let entry_block = state.entry_func.get_entry_block(ctx);
            OpInserter::new_at_block_end(entry_block)
        };
        Self {
            ctx: CtxHandle::Rc(ctx),
            inserter: InserterHandle::owned(inserter),
        }
    }

    /// Create a parse scope that only exists for type registration in the kernel builder.
    /// The state will be incomplete and should never be used for actual codegen.
    pub fn dummy() -> Self {
        let ctx = dummy_context();
        let inserter = {
            let ctx = unsafe { &*ctx.get() };
            let state = ctx.aux_ty::<GlobalState>();
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
    pub fn create_local_mut(&self, value_ty: impl Into<TypeHandle>) -> Value {
        let value_ty = value_ty.into();
        let ctx = self.ctx_mut();
        let align = value_ty.align(ctx);
        let op = DeclareVariableOp::new(ctx, TypeAttr::new(value_ty), AddressSpace::Local, align);
        let out = op.get_result(ctx);
        self.inserter().append_op(ctx, &op);
        out
    }

    /// Create a shared variable of the given item type.
    pub fn create_shared(
        &self,
        value_ty: impl Into<TypeHandle>,
        alignment: Option<usize>,
    ) -> Value {
        let value_ty = value_ty.into();
        let ctx = self.ctx_mut();
        let align = alignment.unwrap_or_else(|| value_ty.align(ctx));
        let op = DeclareVariableOp::new(ctx, TypeAttr::new(value_ty), AddressSpace::Local, align);
        let out = op.get_result(ctx);
        self.inserter().append_op(ctx, &op);
        out
    }

    /// Create a new function.
    pub fn create_function(&self, func: FuncOp) -> usize {
        // We know state doesn't overlap with the stuff that's used in `append_op` so this is safe
        let ctx = self.ctx();
        let state = self.state_mut();
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
        self.state_mut().register_type::<T>(elem);
    }

    /// Register the comptime size for the given generic size.
    pub fn register_size<T: 'static>(&self, size: usize) {
        let state = self.state_mut();

        state.sizemap.insert(TypeId::of::<T>(), size);
    }

    /// Register the type and size of a scalarizable type
    pub fn register_value_type<T: 'static, N: 'static>(&self, value: impl Typed) {
        let ty = value.get_type(self.ctx());
        let scalar_ty = ty.scalar_ty(self.ctx());
        let vector_size = ty.vector_size(self.ctx());
        let storage_ty = {
            let ctx = self.ctx();
            let scalar_ty = scalar_ty.deref(ctx);
            let scalar = type_cast::<dyn ScalarType>(&*scalar_ty).unwrap();
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
    pub fn global(
        &self,
        buffer_pos: usize,
        ext_meta_pos: Option<usize>,
        value_ty: TypeHandle,
    ) -> Value {
        let entry_func = self.state().entry_func;
        let ctx = self.ctx_mut();

        let ty_arr = RuntimeArrayType::get(ctx, value_ty);
        let ty = PointerType::get(ctx, ty_arr.into(), AddressSpace::Global(buffer_pos));

        let id = entry_func.push_argument(ctx, ty.to_handle());
        entry_func.set_arg_attr(
            ctx,
            id,
            &ATTR_BUFFER_BINDING,
            Box::new(BufferBindingAttr::new(buffer_pos, ext_meta_pos)),
        );

        entry_func.get_entry_block(ctx).deref(ctx).get_argument(id)
    }

    /// Obtain the index-th tensor map
    pub fn tensor_map(&self) -> Value {
        let entry_func = self.state().entry_func;
        let ctx = self.ctx();
        let ty = TensorMapType::get(ctx);

        let id = entry_func.push_argument(ctx, ty.to_handle());
        entry_func.get_entry_block(ctx).deref(ctx).get_argument(id)
    }

    pub fn kernel_arg(&self, idx: usize) -> Value {
        let entry_block = self.state().entry_func.get_entry_block(self.ctx());
        entry_block.deref(self.ctx()).get_argument(idx)
    }

    pub fn extract_field(&self, aggregate: Value, field: usize) -> Value {
        let ctx = self.ctx_mut();
        let op = AggregateExtractOp::new(ctx, aggregate, field);
        self.register(&op);
        op.get_result(ctx)
    }

    pub fn const_usize(&self, value: usize) -> Value {
        let op = ConstantOp::new(self.ctx_mut(), IndexAttr::new(value).into());
        self.register(&op);
        op.get_result(self.ctx())
    }

    pub fn into_context(self) -> Option<Context> {
        match self.ctx {
            CtxHandle::Rc(ctx) => Some(Rc::into_inner(ctx)?.into_inner()),
            CtxHandle::Ref(_) => None,
        }
    }
}

impl Display for Scope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let ctx = self.ctx();
        let state = self.state();
        write!(f, "{}", state.module.disp(ctx))
    }
}
