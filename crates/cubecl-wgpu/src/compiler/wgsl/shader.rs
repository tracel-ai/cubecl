use core::fmt::{self, Display, Write};

use cubecl_core::prelude::Visibility;
use cubecl_ir::{
    AddressSpace, GlobalState,
    attributes::{
        ATTR_BUFFER_BINDING, ATTR_READ_WRITE, ATTR_READONLY, ATTR_WRITEONLY, BufferBindingAttr,
        EntrypointInterface, FuncInterface,
    },
    dialect::{BlockPtrExt, memory::AddressSpaceAttr},
    ident,
    interfaces::TypedExt,
    prelude::*,
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
};

use crate::compiler::wgsl::{
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
}

#[op_interface_impl]
impl OpToWgsl for GlobalVariableOp {
    fn to_wgsl(&self, ctx: &Context) -> String {
        let name = self.get_symbol_name(ctx);
        let ty = self.value_ty(ctx).get_type(ctx).to_wgsl(ctx);
        let op = self.get_operation().deref(ctx);
        let attrs = &op.attributes;
        let addr_space = match self.address_space(ctx).0 {
            AddressSpace::Global(_) => {
                if attrs.0.contains_key(&*ATTR_READONLY) {
                    "storage, read"
                } else {
                    "storage, read_write"
                }
            }
            AddressSpace::Shared => "workgroup",
            AddressSpace::Local => "function",
        };
        if let Some(BufferBindingAttr { buffer_pos, .. }) =
            attrs.get::<BufferBindingAttr>(&ATTR_BUFFER_BINDING)
        {
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
pub struct AddressOfOp {
    variable: IdentifierAttr,
}

wgsl_op_with_out!(AddressOfOp, |op, ctx| {
    let name: Identifier = op.variable(ctx).clone().into();
    format!("&{name}")
});

pub fn rewrite_args(ctx: &mut Context, func: FuncOp) -> Vec<Visibility> {
    let module = func.get_operation().parent_module(ctx).get_body(ctx, 0);
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.set_insertion_point_to_block_start(func.get_entry_block(ctx));

    let mut buffers = vec![];
    let args = func.get_entry_block(ctx).arguments(ctx);

    // Back to front so indices don't shift when args get removed
    for (i, &arg) in args.iter().enumerate().rev() {
        let name = arg.unique_name(ctx);
        let attrs = func
            .get_arg_attrs(ctx, i)
            .map(|it| it.clone().0.0)
            .unwrap_or_default();
        if attrs.contains_key(&*ATTR_READONLY) {
            buffers.insert(0, Visibility::Read);
        } else if attrs.contains_key(&*ATTR_READ_WRITE) || attrs.contains_key(&*ATTR_WRITEONLY) {
            buffers.insert(0, Visibility::ReadWrite);
        } else if attrs.contains_key(&ATTR_BUILTIN) {
            continue;
        } else {
            panic!("Should have visibility or builtin annotation")
        }
        let value_ty = arg.get_type(ctx).unwrap_ptr(ctx);
        let var = GlobalVariableOp::new(ctx, value_ty, AddressSpace::Global(i));
        var.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .extend(attrs);
        var.set_symbol_name(ctx, name.clone());
        var.get_operation().insert_at_front(module, ctx);

        if !cfg!(exclusive_memory_only) {
            var.get_operation()
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&ATTR_READONLY);
        }

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
    fn required_feature(&self, ctx: &Context) -> String;
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
                feats.insert(op.required_feature(ctx));
            },
        );

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

pub struct ComputeShader {
    pub buffers: Vec<Visibility>,
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
        let (x, y, z) = entry.cube_dim.into();
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

// impl Display for ComputeShader {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         // On wasm, writeout what extensions we're using. This is standard wgsl but not yet
//         // supported by wgpu.
//         if self.subgroup_instructions_used {
//             #[cfg(target_family = "wasm")]
//             f.write_str("enable subgroups;")?;
//         }

//         if self.f16_used {
//             f.write_str("enable f16;")?;
//         }

//         for value in self.shared_values.iter() {
//             let location = "workgroup";
//             write!(
//                 f,
//                 "var<{location}> {}_store: {};\n\n",
//                 value.value, value.ty,
//             )?;
//         }

//         for extension in self.extensions.iter() {
//             write!(f, "{extension}\n\n")?;
//         }

//         Ok(())
//     }
// }
