use alloc::{boxed::Box, format, rc::Rc, string::String, vec, vec::Vec};
use core::{
    any::{TypeId, type_name},
    cell::{Ref, RefCell, RefMut, UnsafeCell},
    fmt::{Debug, Display},
    sync::atomic::Ordering,
};
use cubecl_common::{format::type_name_sanitized, stub::Mutex};
use derive_more::{Eq, PartialEq};
use enumset::EnumSet;
use hashbrown::HashMap;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::{
        attributes::{TypeAttr, VecAttr},
        op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
        ops::{ConstantOp, FuncOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, UnitType},
    },
    context::{AuxDataIndex, Context},
    debug_info::set_operation_result_name,
    dict_key,
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
use portable_atomic::AtomicUsize;
use spin::LazyLock;

use crate::{
    AddressSpace, AddressType, DeviceProperties, ElemType, FastMath, TargetProperties, TypeHash,
    arena::DropBump,
    attributes::{
        ATTR_BUFFER_BINDING, ATTR_KEY_ARG_ATTRS, ATTR_TENSOR_MAP_BINDING, BoolAttr,
        BufferBindingAttr, EntrypointAbiAttr, EntrypointInterface, FuncInterface, IndexAttr,
    },
    dialect::{
        branch::{IfOp, ReturnOp, YieldOp},
        general::AggregateExtractOp,
        memory::DeclareVariableOp,
    },
    interfaces::{ScalarType, TypedExt},
    read_value,
    settings::KernelSettings,
    types::{PointerType, RuntimeArrayType, cuda::TensorMapType, scalar::BoolType},
};

pub type Types = HashMap<TypeId, ElemType>;
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
    expand_state: RefCell<ExpandState>,
}

#[derive(Clone, Copy, Default)]
pub struct ExpandState {
    pub may_return: bool,
    pub may_break: bool,
    // Whether the kernel has *not* returned. Inverted to save a not on the loop condition.
    pub inv_return_flag: Option<Value>,
    /// Whether the loop is *not* broken. Inverted to save a not on the loop condition.
    pub inv_break_flag: Option<Value>,
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
    pub ident_unique_id: AtomicUsize,
    pub typemap: Types,
    pub sizemap: Sizes,
    pub modes: InstructionModes,
    pub target_properties: TargetProperties,
    pub device_properties: Option<Rc<DeviceProperties>>,
}

impl GlobalState {
    /// Register the element type for the given generic type.
    pub fn register_type<T: 'static>(&mut self, elem: ElemType) {
        self.typemap.insert(TypeId::of::<T>(), elem);
    }
}

dict_key!(ADDRESS_TYPE_KEY, "kernel_address_type");

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
    fn set_address_type(&mut self, addr: AddressType);
    fn address_type(&self) -> AddressType;
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

    fn set_address_type(&mut self, addr: AddressType) {
        if let Some(key) = self.aux_data_map.get(&*ADDRESS_TYPE_KEY).copied() {
            *self.aux_data.get_mut(key).unwrap() = Box::new(addr);
        } else {
            let key = self.aux_data.insert(Box::new(addr));
            self.aux_data_map.insert(ADDRESS_TYPE_KEY.clone(), key);
        }
    }

    fn address_type(&self) -> AddressType {
        let key = self.aux_data_map[&*ADDRESS_TYPE_KEY];
        *self.aux_data[key].downcast_ref::<AddressType>().unwrap()
    }
}

pub trait FuncOpExt {
    fn push_argument(&self, ctx: &Context, ty: TypeHandle) -> usize;
    fn pop_argument(&self, ctx: &Context);
    fn remove_argument(&self, ctx: &Context, arg_idx: usize);
    fn return_type(&self, ctx: &Context) -> TypeHandle;
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
        let new_func_ty = FunctionType::get(ctx, arg_types, res_types).to_handle();
        self.set_attr_func_type(ctx, new_func_ty.into());
        id
    }

    fn pop_argument(&self, ctx: &Context) {
        let last_idx = self.get_entry_block(ctx).deref(ctx).get_num_arguments() - 1;
        BasicBlock::pop_argument(self.get_entry_block(ctx), ctx);

        let (mut arg_types, res_types) = {
            let current_func_ty = self.get_type(ctx).deref(ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            (current_func_ty.arg_types(), current_func_ty.res_types())
        };

        arg_types.pop();
        let new_func_ty = FunctionType::get(ctx, arg_types, res_types).to_handle();
        self.set_attr_func_type(ctx, new_func_ty.into());
        let mut op = self.get_operation().deref_mut(ctx);
        let arg_attrs = op.attributes.0.get_mut(&ATTR_KEY_ARG_ATTRS);
        if let Some(arg_attrs) = arg_attrs.and_then(|attr| attr.downcast_mut::<VecAttr>()) {
            arg_attrs.0.truncate(last_idx);
        }
    }

    fn remove_argument(&self, ctx: &Context, arg_idx: usize) {
        BasicBlock::remove_argument(self.get_entry_block(ctx), ctx, arg_idx);

        let (mut arg_types, res_types) = {
            let current_func_ty = self.get_type(ctx).deref(ctx);
            let current_func_ty = current_func_ty.downcast_ref::<FunctionType>().unwrap();
            (current_func_ty.arg_types(), current_func_ty.res_types())
        };

        arg_types.remove(arg_idx);
        let new_func_ty = FunctionType::get(ctx, arg_types, res_types).to_handle();
        self.set_attr_func_type(ctx, new_func_ty.into());

        let mut op = self.get_operation().deref_mut(ctx);
        let arg_attrs = op.attributes.0.get_mut(&ATTR_KEY_ARG_ATTRS);
        if let Some(arg_attrs) = arg_attrs.and_then(|attr| attr.downcast_mut::<VecAttr>()) {
            arg_attrs.0.remove(arg_idx);
        }
    }

    fn return_type(&self, ctx: &Context) -> TypeHandle {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<FunctionType>().unwrap().res_types()[0]
    }
}

fn new_context(settings: KernelSettings) -> Rc<UnsafeCell<Context>> {
    let mut ctx = Context::default();
    ctx.set_address_type(settings.address_type);

    let module = ModuleOp::new(&mut ctx, ident("kernel"));
    let module_block = module.get_body(&ctx, 0);
    let mut module_inserter = OpInserter::new_at_block_end(module_block);

    // Start out empty and fill in once args register themselves
    let entry_func_ty = FunctionType::get(&ctx, vec![], vec![UnitType::get(&ctx).into()]);
    let entry_name = ident(settings.kernel_name);
    let abi = EntrypointAbiAttr::new(settings.cube_dim, settings.cluster_dim);
    let entry_func = FuncOp::new(&mut ctx, entry_name, entry_func_ty);
    entry_func.set_entrypoint_abi(&mut ctx, abi);
    module_inserter.append_op(&ctx, &entry_func);

    let mut state = GlobalState {
        reference_arena: Default::default(),
        module,
        module_inserter,
        entry_func,
        ident_unique_id: Default::default(),
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
        ident_unique_id: Default::default(),
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

    #[track_caller]
    pub fn state(&self) -> &GlobalState {
        self.ctx().aux_ty()
    }

    #[track_caller]
    pub fn state_mut(&self) -> &mut GlobalState {
        self.ctx_mut().aux_ty_mut()
    }

    fn ident_id(&self) -> usize {
        self.state().ident_unique_id.fetch_add(1, Ordering::SeqCst)
    }

    #[track_caller]
    pub fn expand_state(&self) -> Ref<'_, ExpandState> {
        self.expand_state.borrow()
    }

    #[track_caller]
    pub fn expand_state_mut(&self) -> RefMut<'_, ExpandState> {
        self.expand_state.borrow_mut()
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
        let mut inserter = {
            let ctx = unsafe { &*ctx.get() };
            let state = ctx.aux_ty::<GlobalState>();
            let entry_block = state.entry_func.get_entry_block(ctx);
            OpInserter::new_at_block_end(entry_block)
        };
        let return_flag =
            init_bool_flag(unsafe { &mut *ctx.get() }, &mut inserter, "inv_return_flag");
        Self {
            ctx: CtxHandle::Rc(ctx),
            inserter: InserterHandle::owned(inserter),
            expand_state: RefCell::new(ExpandState {
                may_break: false,
                may_return: false,
                inv_return_flag: Some(return_flag),
                inv_break_flag: None,
            }),
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
            expand_state: Default::default(),
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
            expand_state: RefCell::new(ExpandState {
                may_return: false,
                may_break: false,
                inv_return_flag: None,
                inv_break_flag: None,
            }),
        }
    }

    /// Create a new mutable local variable of type specified by `value_ty`.
    /// `initializer` is a constant attribute and has the same rules as `OpConstant`. This is because
    /// SPIR-V does not allow non-constant (technically non-global, but constants are the only
    /// non-pointer globals) initializers.
    pub fn create_local_mut(
        &self,
        value_ty: impl Into<TypeHandle>,
        init: Option<AttrObj>,
    ) -> Value {
        let value_ty = value_ty.into();
        let ctx = self.ctx_mut();
        let align = value_ty.align(ctx);
        let op = DeclareVariableOp::new(ctx, value_ty, AddressSpace::Local, align, init);
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
        let op = DeclareVariableOp::new(ctx, value_ty, AddressSpace::Shared, align, None);
        let out = op.get_result(ctx);
        self.inserter().append_op(ctx, &op);
        out
    }

    pub fn func_ident(&self, label: Option<&str>) -> Identifier {
        let unique_id = self.ident_id();
        match label {
            Some(label) => ident(format!("{label}_{unique_id}")),
            None => ident(format!("func_{unique_id}")),
        }
    }

    /// Create a new function.
    pub fn register_func(&self, func: FuncOp) {
        let ctx = self.ctx();
        let state = self.state_mut();
        state.module_inserter.append_op(ctx, &func);
    }

    /// Register an [`Instruction`] into the scope.
    pub fn register(&self, op: &dyn Op) {
        let ctx = self.ctx();
        self.inserter().append_op(ctx, op);
    }

    /// Register an [`Instruction`] into the scope and return its result.
    pub fn register_with_result(&self, op: &dyn OneResultInterface) -> Value {
        self.register(op);
        op.get_result(self.ctx())
    }

    /// Terminate block with a `cube.yield` if not already terminated
    pub fn terminate_yield(&self) {
        let block = self.inserter().get_insertion_block(self.ctx());
        let block = block.expect("Should have insertion block");
        if block.deref(self.ctx()).get_terminator(self.ctx()).is_none() {
            self.register(&YieldOp::new(self.ctx_mut()));
        }
    }

    pub fn set_break_return(&self, children: &[Scope]) {
        self.set_may_break(children);
        self.set_may_return(children);
    }

    pub fn set_may_return(&self, children: &[Scope]) {
        let child_may_return = children.iter().any(|scope| scope.expand_state().may_return);
        if child_may_return {
            self.expand_state_mut().may_return = true;
            let flag = self.expand_state().inv_return_flag;
            self.predicate_on_flag(flag.expect("Can't return in rewrite context"));
        }
    }

    pub fn set_may_break(&self, children: &[Scope]) {
        let child_may_return = children.iter().any(|scope| scope.expand_state().may_break);
        if child_may_return {
            self.expand_state_mut().may_break = true;
            let flag = self.expand_state().inv_break_flag;
            self.predicate_on_flag(flag.expect("Should have break flag"));
        }
    }

    fn predicate_on_flag(&self, flag: Value) {
        let ctx = self.ctx_mut();
        let cond = read_value(self, flag);
        let predication = IfOp::new(ctx, cond);
        let then_block = predication.then_block(ctx);
        let else_block = predication.else_block(ctx);
        let yield_ = YieldOp::new(ctx).get_operation();
        yield_.insert_at_back(then_block, ctx);
        let yield_ = YieldOp::new(ctx).get_operation();
        yield_.insert_at_back(else_block, ctx);
        self.register(&predication);
        self.inserter()
            .set_insertion_point_to_block_start(then_block);
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
    pub fn resolve_type<T: 'static>(&self) -> Option<ElemType> {
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
    pub fn register_type<T: 'static>(&self, elem: ElemType) {
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
            scalar.elem_type(ctx)
        };
        self.register_type::<T>(storage_ty);
        self.register_size::<N>(vector_size);
    }

    /// Create an empty child scope.
    pub fn child(&self, inserter: impl Inserter + 'static) -> Self {
        Self {
            ctx: self.ctx.clone(),
            inserter: InserterHandle::owned(inserter),
            expand_state: RefCell::new(ExpandState {
                may_break: false,
                may_return: false,
                inv_return_flag: self.expand_state().inv_return_flag,
                inv_break_flag: self.expand_state().inv_break_flag,
            }),
        }
    }

    /// Create a child scope with a new break condition.
    pub fn loop_child(&self, inserter: impl Inserter + 'static) -> Self {
        let break_flag = init_bool_flag(self.ctx_mut(), self.inserter(), "inv_break_flag");
        Self {
            ctx: self.ctx.clone(),
            inserter: InserterHandle::owned(inserter),
            expand_state: RefCell::new(ExpandState {
                may_return: false,
                may_break: false,
                inv_return_flag: self.expand_state().inv_return_flag,
                inv_break_flag: Some(break_flag),
            }),
        }
    }

    /// Create a child that's at the root of a new function
    pub fn func_child(&self, mut inserter: impl Inserter + 'static) -> Self {
        let return_flag = init_bool_flag(self.ctx_mut(), &mut inserter, "inv_return_flag");
        Self {
            ctx: self.ctx.clone(),
            inserter: InserterHandle::owned(inserter),
            expand_state: RefCell::new(ExpandState {
                may_break: false,
                may_return: false,
                inv_return_flag: Some(return_flag),
                inv_break_flag: None,
            }),
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
    pub fn tensor_map(&self, buffer_pos: usize, ext_meta_pos: usize) -> Value {
        let entry_func = self.state().entry_func;
        let ctx = self.ctx();
        let ty = TensorMapType::get(ctx);

        let id = entry_func.push_argument(ctx, ty.to_handle());
        entry_func.set_arg_attr_unit(ctx, id, &ATTR_TENSOR_MAP_BINDING);
        entry_func.set_arg_attr(
            ctx,
            id,
            &ATTR_BUFFER_BINDING,
            Box::new(BufferBindingAttr::new(buffer_pos, Some(ext_meta_pos))),
        );
        entry_func.get_entry_block(ctx).deref(ctx).get_argument(id)
    }

    pub fn kernel_arg(&self, idx: usize) -> Value {
        let entry_block = self.state().entry_func.get_entry_block(self.ctx());
        entry_block.deref(self.ctx()).get_argument(idx)
    }

    pub fn extract_field(&self, aggregate: Value, field: usize) -> Value {
        let ctx = self.ctx_mut();
        let op = AggregateExtractOp::new(ctx, aggregate, field);
        self.register_with_result(&op)
    }

    pub fn const_usize(&self, value: usize) -> Value {
        let op = ConstantOp::new(self.ctx_mut(), IndexAttr::new(value).into());
        self.register_with_result(&op)
    }

    pub fn const_bool(&self, value: bool) -> Value {
        let op = ConstantOp::new(self.ctx_mut(), BoolAttr::new(value).into());
        self.register_with_result(&op)
    }

    pub fn into_context(self) -> Option<Context> {
        let entry = self.state().entry_func.get_entry_block(self.ctx());
        if entry.deref(self.ctx()).get_terminator(self.ctx()).is_none() {
            self.inserter().set_insertion_point_to_block_end(entry);
            self.register(&ReturnOp::new(self.ctx_mut()));
        }
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

fn init_bool_flag(ctx: &mut Context, inserter: &mut dyn Inserter, name: &str) -> Value {
    let bool = TypeAttr::new(BoolType::get(ctx).to_handle());
    let false_ = BoolAttr::new(true).into();
    let flag = DeclareVariableOp::new(ctx, bool, AddressSpace::Local, 1, Some(false_));
    inserter.append_op(ctx, &flag);
    set_operation_result_name(ctx, flag.get_operation(), 0, Some(ident(name)));
    flag.get_result(ctx)
}
