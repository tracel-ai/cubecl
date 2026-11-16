use core::fmt::{self, Display, Write};

use cubecl_core::prelude::Visibility;
use cubecl_ir::{
    AddressSpace, CanMaterialize, GlobalState, Pure,
    attributes::{
        ATTR_BUFFER_BINDING, ATTR_BUFFER_IO, BufferBindingAttr, BufferIOAttr, EntrypointInterface,
        FuncInterface,
    },
    dialect::{BlockPtrExt, memory::AddressSpaceAttr},
    ident,
    interfaces::TypedExt,
    prelude::*,
    typed_vec_attr,
    types::StructType,
};
use hashbrown::HashSet;
use itertools::Itertools;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IdentifierAttr,
        ops::{FuncOp, ModuleOp},
        types::UnitType,
    },
    common_traits::Named,
    identifier::Identifier,
    irbuild::listener::DummyListener,
    linked_list::ContainsLinkedList,
    utils::table::ISet,
};

#[cfg(target_family = "wasm")]
use crate::compiler::wgsl::builtin::BuiltIn;
use crate::compiler::wgsl::{
    KernelInfo,
    builtin::{ATTR_BUILTIN, BuiltInAttr},
    to_wgsl::{OpExtWgsl, OpToWgsl, TypeExtWgsl, wgsl_op, wgsl_op_with_out},
    value::WgslValue,
};

#[cube_op(name = "wgsl.global_variable", format = "attr_dict")]
#[result_ty(none)]
#[derive_op_interface_impl(SymbolOpInterface)]
pub struct GlobalVariableOp {
    value_ty: TypeAttr,
    address_space: AddressSpaceAttr,
    #[attribute(optional)]
    buffer_binding: BufferBindingAttr,
    #[attribute(optional)]
    buffer_io: BufferIOAttr,
}

#[op_interface_impl]
impl OpToWgsl for GlobalVariableOp {
    fn to_wgsl(&self, ctx: &Context) -> String {
        let name = self.get_symbol_name(ctx);
        let ty = self.value_ty(ctx).get_type(ctx).to_wgsl(ctx);
        let addr_space = match self.address_space(ctx).0 {
            AddressSpace::Global(_) => {
                let io = *self.buffer_io(ctx).expect("Should have IO");
                if io.is_writable() {
                    "storage, read_write"
                } else {
                    "storage, read"
                }
            }
            AddressSpace::Shared => "workgroup",
            AddressSpace::Local => "function",
        };
        if let Some(BufferBindingAttr { buffer_pos, .. }) = self.buffer_binding(ctx).map(|it| *it) {
            format!("@group(0) @binding({buffer_pos}) var<{addr_space}> {name}: {ty};\n")
        } else {
            format!("var<{addr_space}> {name}: {ty};\n")
        }
    }
}

#[cube_op(
    name = "wgsl.address_of",
    format = "`@` attr($variable, $IdentifierAttr) ` : ` type($0)"
)]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct AddressOfOp {
    variable: IdentifierAttr,
}

wgsl_op_with_out!(AddressOfOp; |op, ctx| {
    let name: Identifier = op.variable(ctx).clone().into();
    format!("&{name}")
});

pub fn rewrite_args(ctx: &mut Context, func: FuncOp) -> Vec<Visibility> {
    let entry = func.get_operation();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.set_insertion_point_to_block_start(func.get_entry_block(ctx));

    let mut buffers = vec![];
    let args = func.get_entry_block(ctx).arguments(ctx);

    // Back to front so indices don't shift when args get removed
    for (i, &arg) in args.iter().enumerate().rev() {
        let name = arg.unique_name(ctx);
        let binding = {
            let Some(binding) =
                func.get_arg_attr::<BufferBindingAttr>(ctx, i, &ATTR_BUFFER_BINDING)
            else {
                continue;
            };
            *binding
        };
        let mut io = {
            let Some(io) = func.get_arg_attr::<BufferIOAttr>(ctx, i, &ATTR_BUFFER_IO) else {
                panic!("Should have visibility or builtin annotation")
            };
            *io
        };

        if !cubecl_server::memory_management::EXCLUSIVE_MEMORY_ONLY {
            io = BufferIOAttr::ReadWrite;
        }

        if io.is_writable() {
            buffers.insert(0, Visibility::ReadWrite);
        } else {
            buffers.insert(0, Visibility::Read);
        }

        let value_ty = arg.get_type(ctx).unwrap_ptr(ctx);
        let var = GlobalVariableOp::new(
            ctx,
            value_ty,
            AddressSpace::Global(i),
            Some(binding),
            Some(io),
        );
        var.set_symbol_name(ctx, name.clone());
        var.get_operation().insert_before(ctx, entry);

        let addr = AddressOfOp::new(ctx, arg.get_type(ctx), name);
        rewriter.append_op(ctx, &addr);
        rewriter.replace_value_uses_with(ctx, arg, addr.get_result(ctx));
        func.remove_argument(ctx, i);
    }

    buffers
}

pub fn shared_memory_size(ctx: &Context, op: Ptr<Operation>) -> usize {
    let mut size = 0;
    visit_all_ops_of_type::<GlobalVariableOp, _>(ctx, &mut size, op, |ctx, size, op| {
        if matches!(op.address_space(ctx).0, AddressSpace::Shared) {
            *size += op.value_ty(ctx).size(ctx);
        }
    });
    size
}

#[cube_op(name = "wgsl.enable", format = "attr($feature, $IdentifierAttr)")]
#[result_ty(none)]
pub struct EnableOp {
    feature: IdentifierAttr,
}

wgsl_op!(EnableOp, |op, ctx| {
    format!("enable {};\n", op.feature(ctx).as_ref())
});

#[op_interface]
pub trait RequiresFeatureOp {
    verify_op_succ!();
    fn required_feature(&self, ctx: &Context) -> Vec<String>;
}

pub struct EnableFeaturesPass;

#[pass_name]
impl Pass for EnableFeaturesPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let module_body = op.as_op::<ModuleOp>(ctx).unwrap().get_body(ctx, 0);
        let mut feats = HashSet::new();
        visit_all_values(ctx, &mut feats, op, |ctx, feats, val| {
            if let Some(elem) = val.try_get_scalar_elem_ty(ctx)
                && elem.is_float16(ctx)
            {
                feats.insert("f16".to_string());
            }
        });
        visit_all_ops_with_interface::<dyn RequiresFeatureOp, _>(
            ctx,
            &mut feats,
            op,
            |ctx, feats, op| {
                feats.extend(op.required_feature(ctx));
            },
        );
        // Browser WebGPU requires a subgroup enable directive for builtins too.
        // Native Naga supports subgroup use without the directive and rejects it.
        #[cfg(target_family = "wasm")]
        visit_all_ops_of_type::<FuncOp, _>(ctx, &mut feats, op, |ctx, feats, op| {
            for i in 0..op.get_entry_block(ctx).arguments(ctx).len() {
                if let Some(builtin) = op.get_arg_attr::<BuiltInAttr>(ctx, i, &ATTR_BUILTIN)
                    && matches!(
                        builtin.0,
                        BuiltIn::SubgroupSize | BuiltIn::SubgroupId | BuiltIn::SubgroupInvocationId
                    )
                {
                    feats.insert("subgroups".to_string());
                }
            }
        });

        let mut res = PassResult::default();
        if !feats.is_empty() {
            res.ir_changed = IRStatus::Changed;
        }

        for feat in feats {
            let enable = EnableOp::new(ctx, ident(feat));
            enable.get_operation().insert_at_front(module_body, ctx);
        }
        Ok(res)
    }
}

#[format]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct StructField {
    ty: TypeHandle,
    size: u64,
}

typed_vec_attr!(StructField, "wgsl.fields", StructFieldsAttr);

#[cube_op(
    name = "wgsl.declare_struct",
    format = "attr($fields, $StructFieldsAttr)"
)]
#[result_ty(none)]
pub struct DeclareStructOp {
    name: IdentifierAttr,
    fields: StructFieldsAttr,
}

#[op_interface_impl]
impl RequiresFeatureOp for DeclareStructOp {
    fn required_feature(&self, ctx: &Context) -> Vec<String> {
        self.fields(ctx)
            .0
            .iter()
            .filter_map(|field| {
                if let Some(ty) = field.ty.try_get_scalar_elem_ty(ctx)
                    && ty.is_float16(ctx)
                {
                    return Some("f16".to_string());
                }
                None
            })
            .collect()
    }
}

wgsl_op!(DeclareStructOp, |op, ctx| {
    let name = op.name(ctx);
    let fields = op.fields(ctx);
    let fields = fields.0.iter().enumerate();
    let fields = fields
        .map(|(i, field)| {
            let ty = field.ty.to_wgsl(ctx);
            format!("  @size({}) _{i}: {ty},", field.size)
        })
        .join("\n");
    format!("struct {} {{\n{fields}\n}};\n", name.as_ref())
});

#[derive(Default)]
pub struct DeclareStructTypesPass;

#[pass_name]
impl Pass for DeclareStructTypesPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let module = op.as_op::<ModuleOp>(ctx).expect("Should be run on module");
        let mut structs = ISet::default();

        visit_all_values(ctx, &mut structs, op, |ctx, structs, val| {
            visit_structs_nested(ctx, val, structs);
        });

        let mut res = PassResult::default();
        for r#struct in structs {
            let (name, fields) = {
                let r#struct = r#struct.deref(ctx);
                let name = format!("Struct_{:x}", r#struct.stable_hash());
                let fields = r#struct
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, field)| field.ty.size(ctx) > 0)
                    .map(|(i, field)| {
                        let next_offs = r#struct.fields.get(i + 1).map(|it| it.offset);
                        let next_offs = next_offs.unwrap_or(r#struct.size);
                        let size = next_offs - field.offset;
                        StructField { size, ty: field.ty }
                    });
                (name, fields.collect())
            };
            let decl = DeclareStructOp::new(ctx, ident(name), StructFieldsAttr(fields));
            decl.get_operation()
                .insert_at_front(module.get_body(ctx, 0), ctx);
            res.ir_changed |= IRStatus::Changed;
        }
        Ok(res)
    }
}

fn visit_structs_nested(
    ctx: &Context,
    val: impl Typed,
    structs: &mut ISet<TypedHandle<StructType>>,
) {
    if let Some(elem) = val.try_get_scalar_elem_ty(ctx)
        && let Ok(struct_ty) = TypedHandle::<StructType>::from_handle(elem, ctx)
    {
        if structs.contains(&struct_ty) {
            return;
        }
        for field in struct_ty.deref(ctx).fields.iter() {
            visit_structs_nested(ctx, field.ty, structs);
        }
        structs.insert(struct_ty);
    }
}

pub struct ComputeShader {
    pub buffers: Vec<Visibility>,
    /// What the kernel does with each buffer binding, by buffer position —
    /// the four-state answer the launch path's taint bookkeeping consumes.
    /// Captured from the IR attributes before [`rewrite_args`] widens them
    /// for the shader: the shader's visibility is deliberately forced wider
    /// than the kernel's own behavior, and the taint bookkeeping needs the
    /// kernel's, not the shader's.
    pub io: Vec<BufferIOAttr>,
    pub shared_memory_size: usize,
    pub ctx: Context,
}

impl Display for ComputeShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let module = self.ctx.aux_ty::<GlobalState>().module;
        write!(f, "{}", module.to_wgsl(&self.ctx))
    }
}

wgsl_op!(ModuleOp, |op, ctx| {
    block_to_wgsl(ctx, op.get_body(ctx, 0))
});

wgsl_op!(FuncOp, |op, ctx| func_to_wgsl(ctx, op).unwrap());

fn func_to_wgsl(ctx: &Context, op: &FuncOp) -> core::result::Result<String, fmt::Error> {
    let mut sig = String::new();
    let f = &mut sig;
    if let Some(entry) = op.get_entrypoint_abi(ctx) {
        let layout = ctx.aux_ty::<KernelInfo>().layout;
        let (x, y, z) = layout.workgroup_size(entry.cube_dim).into();
        writeln!(f, "@compute @workgroup_size({x}, {y}, {z})")?;
    }
    let name = op.get_symbol_name(ctx);
    let entry = op.get_entry_block(ctx);
    let args = entry.arguments(ctx);
    let ret = op.return_type(ctx);

    let mut args = args.iter().enumerate().map(|(i, &arg)| {
        let name = arg.name(ctx);
        let ty = arg.get_type(ctx).to_wgsl(ctx);
        match op.get_arg_attr::<BuiltInAttr>(ctx, i, &ATTR_BUILTIN) {
            Some(builtin) => format!("@builtin({}) {name}: {ty}", builtin.0),
            None => format!("{name}: {ty}"),
        }
    });
    write!(f, "fn {name}({})", args.join(", "))?;
    if !ret.deref(ctx).is::<UnitType>() {
        write!(f, " -> {}", ret.to_wgsl(ctx))?;
    }

    Ok(format!("{sig} {{\n{}\n}}\n", block_to_wgsl(ctx, entry)))
}

pub fn block_to_wgsl(ctx: &Context, block: Ptr<BasicBlock>) -> String {
    let mut out = String::new();
    let ops = block.deref(ctx).iter(ctx);
    for op in ops {
        out.push_str(&op.to_wgsl(ctx).unwrap());
    }
    out
}
