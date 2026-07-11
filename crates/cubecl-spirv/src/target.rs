use core::any::type_name;
use cubecl_core::{
    WgpuCompilationOptions,
    ir::{
        ExpandValue, Scope, UIntKind,
        attributes::{
            ATTR_BUFFER_BINDING, ATTR_READONLY, ATTR_WRITEONLY, BufferBindingAttr,
            EntrypointInterface, FuncInterface,
        },
        ident,
        interfaces::TypedExt,
        metadata::Info,
        prelude::*,
    },
};
use pliron::{
    builtin::ops::FuncOp,
    identifier::Identifier,
    irbuild::{inserter::Inserter, listener::DummyListener},
    printable::Printable,
    std_deps::sync::LazyLock,
};
use pliron_spirv::{
    decorations::{DecoratableOp, DecorationInfo},
    interfaces::VerCapExtInterface,
    ops::{AddressOfOp, GlobalVariableOp, InBoundsAccessChainOp, LoadOp, SpirvModuleOp},
    types::{ArrayType, PointerType, RuntimeArrayType, StructType},
};
use rspirv::spirv::{Capability, Decoration, MemoryAccess, StorageClass};

use crate::types::ty_to_spirv_dialect;

pub static PARAMS_NAME: LazyLock<Identifier> = LazyLock::new(|| ident("_spirv_params"));

/// Run on: `SpirvModuleOp`
pub struct ConvertArgsPass;

impl Pass for ConvertArgsPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut module = op
            .as_op::<SpirvModuleOp>(ctx)
            .expect("Should run on SPIR-V module");
        visit_all_ops_of_type_mut::<FuncOp, _>(ctx, &mut module, op, |ctx, module, func| {
            if func.get_entrypoint_abi(ctx).is_some() {
                Self::convert_func(ctx, *module, func);
            }
        });
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

impl ConvertArgsPass {
    pub fn convert_func(ctx: &mut Context, module: SpirvModuleOp, func: FuncOp) {
        let info = ctx.aux_ty::<Info>().clone();
        let entry = func.get_entry_block(ctx);
        let module_body = module.get_body(ctx, 0);
        let args = entry.deref(ctx).arguments().collect::<Vec<_>>();
        let mut buffers = vec![];

        for (i, arg) in args.iter().enumerate() {
            let binding = func.get_arg_attr::<BufferBindingAttr>(ctx, i, &ATTR_BUFFER_BINDING);
            let non_writable = func.has_arg_attr(ctx, i, &ATTR_READONLY);
            let non_readable = func.has_arg_attr(ctx, i, &ATTR_WRITEONLY);
            if let Some(binding) = binding {
                buffers.insert(
                    binding.buffer_pos,
                    (
                        *arg,
                        buffer_ty(ctx, *arg, module),
                        non_readable,
                        non_writable,
                    ),
                );
            } else {
                panic!("Expected all kernel inputs to be bindings")
            }
        }

        let mut addr_struct = StructType::default();
        addr_struct.decorate_type(DecorationInfo::unit(Decoration::Block));

        for (i, (_, ty, non_readable, non_writable)) in buffers.iter().copied().enumerate() {
            addr_struct.field_types.push(ty);
            addr_struct.offsets.push((i * size_of::<u64>()) as u32);
            if non_readable {
                addr_struct.decorate_member(i, DecorationInfo::unit(Decoration::NonReadable));
            }
            if non_writable {
                addr_struct.decorate_member(i, DecorationInfo::unit(Decoration::NonWritable));
            }
        }

        if info.has_info() {
            let offset = buffers.len() * size_of::<u64>();
            addr_struct.field_types.push(info_ty(ctx, module));
            addr_struct.offsets.push(offset as u32);
            addr_struct
                .decorate_member(buffers.len(), DecorationInfo::unit(Decoration::NonWritable));
        }

        let addr_struct = Type::register_instance(addr_struct, ctx).to_handle();
        let storage_class = params_storage_class(ctx, buffers.len());
        let addr_struct_ptr = PointerType::get(ctx, addr_struct, storage_class).into();
        let ptrs_var =
            GlobalVariableOp::new(ctx, addr_struct, storage_class, PARAMS_NAME.clone(), None);
        ptrs_var.get_operation().insert_at_front(module_body, ctx);

        if !matches!(storage_class, StorageClass::PushConstant) {
            ptrs_var.set_decoration_descriptor_set(ctx, 0.into());
            ptrs_var.set_decoration_binding(ctx, 0.into());
        }

        let mut rewriter = IRRewriter::<DummyListener>::default();
        rewriter.set_insertion_point_to_block_start(entry);
        let scope = Scope::from_context_and_inserter(ctx, &mut rewriter);
        let ptrs = scope.register_with_result(&AddressOfOp::new(
            ctx,
            addr_struct_ptr,
            PARAMS_NAME.clone(),
        ));

        for (i, (buffer, ..)) in buffers.iter().enumerate() {
            load_buffer_array(ctx, &scope, ptrs, i, *buffer, storage_class);
        }

        // TODO
        let _info = info
            .has_info()
            .then(|| load_buffer(ctx, &scope, ptrs, buffers.len(), storage_class));
        let num_args = entry.deref(ctx).get_num_arguments();

        for _ in 0..num_args {
            func.pop_argument(ctx);
        }
    }
}

fn buffer_ty(ctx: &Context, buffer: Value, module: SpirvModuleOp) -> TypeHandle {
    let ty = buffer.element_ty(ctx);
    let ty_size = ty.size(ctx);
    let ty = ty_to_spirv_dialect(ctx, ty);
    match ty_size {
        1 => module.insert_capability(ctx, Capability::StorageBuffer8BitAccess),
        2 => module.insert_capability(ctx, Capability::StorageBuffer16BitAccess),
        _ => {}
    }
    let array = RuntimeArrayType::get(ctx, ty, Some(ty_size as u32));
    let struct_ = StructType::get(
        ctx,
        vec![array.into()],
        vec![0],
        vec![],
        vec![DecorationInfo::unit(Decoration::Block)],
    );
    PointerType::get(ctx, struct_.into(), StorageClass::PhysicalStorageBuffer).into()
}

fn info_ty(ctx: &Context, module: SpirvModuleOp) -> TypeHandle {
    let address_ty = ctx.address_type();
    let info = ctx.aux_ty::<Info>().clone();

    let mut struct_ = StructType::default();
    struct_
        .type_decorations
        .push(DecorationInfo::unit(Decoration::Block));

    for scalar in info.scalars {
        let ty = ty_to_spirv_dialect(ctx, scalar.ty.to_type(ctx));
        let ty_size = scalar.ty.size() as u32;
        match ty_size {
            1 => {
                module.insert_capability(ctx, Capability::StorageBuffer8BitAccess);
                module.insert_capability(ctx, Capability::UniformAndStorageBuffer8BitAccess);
            }
            2 => {
                module.insert_capability(ctx, Capability::StorageBuffer16BitAccess);
                module.insert_capability(ctx, Capability::UniformAndStorageBuffer16BitAccess);
            }
            _ => {}
        }

        let array = ArrayType::get(ctx, scalar.count as u32, ty, Some(ty_size));
        struct_.field_types.push(array.into());
        struct_.offsets.push(scalar.offset as u32);
    }

    if let Some(field) = info.sized_meta {
        let ty = ty_to_spirv_dialect(ctx, field.ty.to_type(ctx));
        let ty_size = field.ty.size() as u32;
        let array = ArrayType::get(ctx, field.count as u32, ty, Some(ty_size));
        struct_.field_types.push(array.into());
        struct_.offsets.push(field.offset as u32);
    }

    if info.has_dynamic_meta {
        let address_ty = ty_to_spirv_dialect(ctx, address_ty.unsigned_type().to_type(ctx));
        let ty_size = address_ty.size(ctx) as u32;
        let array = RuntimeArrayType::get(ctx, address_ty, Some(ty_size));
        struct_.field_types.push(array.into());
        struct_.offsets.push(info.dynamic_meta_offset as u32);
    }

    let struct_ = Type::register_instance(struct_, ctx).to_handle();
    PointerType::get(ctx, struct_, StorageClass::PhysicalStorageBuffer).into()
}

fn load_buffer_array(
    ctx: &mut Context,
    scope: &Scope,
    ptrs: Value,
    idx: usize,
    cube_buffer: Value,
    storage_class: StorageClass,
) {
    let buffer = load_buffer(ctx, scope, ptrs, idx, storage_class);
    let buffer_ty = TypedHandle::<PointerType>::from_handle(buffer.get_type(ctx), ctx).unwrap();
    let buffer_ty =
        TypedHandle::<StructType>::from_handle(buffer_ty.deref(ctx).element_type, ctx).unwrap();
    let array_ty = buffer_ty.deref(ctx).field_types[0];
    let array_ptr_ty = PointerType::get(ctx, array_ty, StorageClass::PhysicalStorageBuffer);

    let zero = ExpandValue::constant(0.into(), UIntKind::U32).value(scope);
    let array = InBoundsAccessChainOp::new(ctx, array_ptr_ty.into(), buffer, vec![zero]);
    let array = scope.register_with_result(&array);
    cube_buffer.replace_all_uses_with(ctx, &array);
}

fn load_buffer(
    ctx: &mut Context,
    scope: &Scope,
    ptrs: Value,
    idx: usize,
    storage_class: StorageClass,
) -> Value {
    let struct_ptr = TypedHandle::<PointerType>::from_handle(ptrs.get_type(ctx), ctx).unwrap();
    let struct_ =
        TypedHandle::<StructType>::from_handle(struct_ptr.deref(ctx).element_type, ctx).unwrap();
    let buffer_ty = struct_.deref(ctx).field_types[idx];
    let buffer_ptr_ty = PointerType::get(ctx, buffer_ty, storage_class);

    let idx = ExpandValue::constant(idx.into(), UIntKind::U32).value(scope);
    let buffer_ptr = InBoundsAccessChainOp::new(ctx, buffer_ptr_ty.into(), ptrs, vec![idx]);
    let buffer_ptr = scope.register_with_result(&buffer_ptr);
    let load = LoadOp::new(ctx, buffer_ty, buffer_ptr, MemoryAccess::NONE, None);
    scope.register_with_result(&load)
}

pub fn params_storage_class(ctx: &Context, num_buffers: usize) -> StorageClass {
    let num_addresses = match ctx.aux_ty::<Info>().has_info() {
        true => num_buffers + 1,
        false => num_buffers,
    };
    let comp_options = &ctx.aux_ty::<WgpuCompilationOptions>().vulkan;
    if num_addresses > comp_options.push_constant_size / size_of::<u64>() {
        StorageClass::Uniform
    } else {
        StorageClass::PushConstant
    }
}

pub struct CollectVerCapExtPass;

impl Pass for CollectVerCapExtPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        let mut module = op.as_op::<SpirvModuleOp>(ctx).unwrap();
        visit_all_ops_with_interface(ctx, &mut module, op, update_ver_cap_ext);
        Ok(res)
    }
}

fn update_ver_cap_ext(ctx: &Context, module: &mut SpirvModuleOp, op: &dyn VerCapExtInterface) {
    let max_ver = ctx
        .aux_ty::<WgpuCompilationOptions>()
        .vulkan
        .max_spirv_version;
    let min_version = op.min_version(ctx);
    let extensions = if min_version.is_some_and(|ver| ver <= max_ver) {
        vec![]
    } else {
        op.required_extensions(ctx)
    };
    let caps = op.required_capabilities(ctx);

    for cap_set in caps.into_iter().filter(|set| !set.is_empty()) {
        if cap_set.len() == 1 {
            module.insert_capability(ctx, cap_set[0]);
        } else if cap_set.iter().any(|cap| module.has_capability(ctx, cap)) {
            continue;
        } else {
            panic!(
                "Need custom rule for multi-capability op {}, lists capabilities: {:?}",
                op.get_operation().disp(ctx),
                cap_set
            );
        }
    }

    for ext_set in extensions.into_iter().filter(|set| !set.is_empty()) {
        if ext_set.len() == 1 {
            module.insert_extension(ctx, ext_set[0]);
        } else if ext_set.iter().any(|ext| module.has_extension(ctx, *ext)) {
            continue;
        } else {
            panic!(
                "Need custom rule for multi-extension op {}, lists extensions: {:?}",
                op.get_operation().disp(ctx),
                ext_set
            );
        }
    }
}
