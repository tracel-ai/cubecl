use cubecl_core::ir::{
    Builtin,
    attributes::EntrypointInterface,
    dialect::{
        general::ReadBuiltinOp,
        synchronization::{SyncOp, SyncScope},
    },
    interfaces::{AlignedType, HasElementType},
    prelude::*,
    types::{ArrayType, PointerType, VectorType},
};
use cubecl_opt::passes::alloc_shared_memory::{AllocSharedOp, SliceSharedOp};
use itertools::Itertools;
use pliron::{
    builtin::{
        op_interfaces::SingleBlockRegionInterface,
        ops::{FuncOp, ModuleOp},
        types::FunctionType,
    },
    graph::walkers::uninterruptible::immutable::walk_op,
    linked_list::ContainsLinkedList,
};

use crate::{
    opencl::{OpenClOptions, opencl_op, opencl_op_with_out, ty::array_name},
    shared::{
        CppValue, OpExtCPP,
        signature::{DeclareVectorOp, IncludeOp},
        ty::TypeExtCPP,
        type_definitions,
    },
};

// Device-wide syncs fence at device scope where the device has it (an OpenCL C 3.0 feature); bits
// are reversed by `cl_khr_extended_bit_ops` or by shifts and masks.
const HEADER: &str = "\
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#if defined(cl_khr_int64_base_atomics) && defined(cl_khr_int64_extended_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#endif
#if defined(__opencl_c_atomic_scope_device) && defined(__opencl_c_atomic_order_acq_rel)
#define sync_device() do { \\
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device); \\
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); \\
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device); \\
} while (0)
#else
#define sync_device() barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)
#endif
#ifdef cl_khr_extended_bit_ops
#pragma OPENCL EXTENSION cl_khr_extended_bit_ops : enable
#define reverse_bits32(x) bit_reverse(x)
#define reverse_bits64(x) bit_reverse(x)
#else
static inline uint reverse_bits32(uint x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
}
static inline ulong reverse_bits64(ulong x) {
    return ((ulong)reverse_bits32((uint)x) << 32) | reverse_bits32((uint)(x >> 32));
}
#endif
";

opencl_op!(ModuleOp, |op, ctx| {
    let mut out = HEADER.to_string();
    type_definitions(&mut out, "long").expect("writing to a String should not fail");
    // Vectors are declared first (by `DeclareVectorTypesPass`, at the front of the module), then
    // the arrays, which may hold vectors, then everything else; the kernels last, after the
    // functions they call (closures are appended to the module after the kernel using them).
    let arrays = declare_arrays(ctx, op.get_operation());
    let mut arrays = Some(arrays);
    let mut kernels = String::new();
    for op in op.get_body(ctx, 0).deref(ctx).iter(ctx) {
        if !op.is_op::<DeclareVectorOp>(ctx)
            && let Some(arrays) = arrays.take()
        {
            out.push_str(&arrays);
        }
        let kernel = op
            .as_op::<FuncOp>(ctx)
            .is_some_and(|func| func.get_entrypoint_abi(ctx).is_some());
        match OpExtCPP::to_cpp(&op, ctx) {
            Ok(cpp) if kernel => kernels.push_str(&cpp),
            Ok(cpp) => out.push_str(&cpp),
            Err(err) => ctx.aux_ty::<crate::error::EmissionErrors>().record(err),
        }
    }
    out.push_str(&arrays.unwrap_or_default());
    out.push_str(&kernels);
    out
});

/// One struct per array type in the module, element types first.
fn declare_arrays(ctx: &Context, module: Ptr<Operation>) -> String {
    let mut arrays = Vec::new();
    walk_op(
        ctx,
        &mut arrays,
        &WALKCONFIG_PREORDER_FORWARD,
        module,
        |ctx, arrays, node| {
            let types: Vec<TypeHandle> = match node {
                IRNode::Operation(op) => {
                    let op = op.deref(ctx);
                    op.results()
                        .chain(op.operands())
                        .map(|value| value.get_type(ctx))
                        .collect()
                }
                IRNode::BasicBlock(block) => block
                    .deref(ctx)
                    .arguments()
                    .map(|arg| arg.get_type(ctx))
                    .collect(),
                _ => Vec::new(),
            };
            for ty in types {
                collect_arrays(ctx, ty, arrays);
            }
        },
    );
    arrays
        .into_iter()
        .map(|(_, declaration)| declaration)
        .collect()
}

/// Whether the module names its subgroups: plane operations, syncs and indices, cooperative matrices.
fn uses_subgroups(ctx: &Context, module: Ptr<Operation>) -> bool {
    let mut found = false;
    walk_op(
        ctx,
        &mut found,
        &WALKCONFIG_PREORDER_FORWARD,
        module,
        |ctx, found, node| {
            if let IRNode::Operation(op) = node {
                *found |= names_subgroups(ctx, op);
            }
        },
    );
    found
}

fn names_subgroups(ctx: &Context, op: Ptr<Operation>) -> bool {
    let dialect = Operation::get_opid(op, ctx).dialect;
    if matches!(dialect.as_ref(), "plane" | "matrix") {
        return true;
    }
    if let Some(sync) = op.as_op::<SyncOp>(ctx) {
        return sync.scope(ctx).0 == SyncScope::Plane;
    }
    op.as_op::<ReadBuiltinOp>(ctx).is_some_and(|read| {
        matches!(
            read.builtin(ctx).0,
            Builtin::PlaneDim | Builtin::PlanePos | Builtin::UnitPosPlane
        )
    })
}

/// Declares the arrays `ty` holds, then `ty` if it is one: once per name (distinct IR types may
/// print alike, as `usize` and the address type's integer).
fn collect_arrays(ctx: &Context, ty: TypeHandle, arrays: &mut Vec<(String, String)>) {
    let ty_ref = ty.deref(ctx);
    let inner = if let Some(ptr) = ty_ref.downcast_ref::<PointerType>() {
        Some(ptr.inner)
    } else if let Some(array) = ty_ref.downcast_ref::<ArrayType>() {
        Some(array.inner)
    } else {
        type_cast::<dyn HasElementType>(&*ty_ref).and_then(|it| it.element_type(ctx))
    };
    if let Some(inner) = inner.filter(|&inner| inner != ty) {
        collect_arrays(ctx, inner, arrays);
    }
    if let Some(array) = ty_ref.downcast_ref::<ArrayType>() {
        let name = array_name(ctx, array);
        if !arrays.iter().any(|(declared, _)| *declared == name) {
            let inner = array.inner.to_cpp(ctx);
            let declaration = format!(
                "typedef struct {{ {inner} data[{}]; }} {name};\n",
                array.length
            );
            arrays.push((name, declaration));
        }
    }
}

opencl_op!(FuncOp, |op, ctx| {
    let func_name = op.get_symbol_name(ctx);
    let ty = op.get_type(ctx).deref(ctx);
    let func_ty = ty
        .downcast_ref::<FunctionType>()
        .expect("a function should have a function type");
    let return_ty = func_ty.res_types()[0].to_cpp(ctx);

    let entry_block = op.get_entry_block(ctx);
    let block = entry_block.deref(ctx);
    let params = block
        .arguments()
        .map(|arg| format!("{} const {}", arg.get_type(ctx).to_cpp(ctx), arg.name(ctx)))
        .join(", ");
    let body = crate::shared::branch::block_to_cpp(ctx, entry_block);

    let attributes = if let Some(abi) = op.get_entrypoint_abi(ctx) {
        let (x, y, z) = abi.cube_dim.into();
        let mut attributes =
            format!("__kernel __attribute__((reqd_work_group_size({x}, {y}, {z})))");
        // A kernel that names its subgroups runs at the subgroup size the device promised for
        // planes; the others at the size the driver picks.
        let options = OpenClOptions::of(ctx);
        let module = op.get_operation().deref(ctx).get_parent_op(ctx);
        if let Some(size) = options.subgroup_size
            && module.is_some_and(|module| uses_subgroups(ctx, module))
        {
            attributes.push_str(&format!(
                " __attribute__((intel_reqd_sub_group_size({size})))"
            ));
        }
        format!("{attributes} {return_ty}")
    } else {
        format!("static inline {return_ty}")
    };

    format!("{attributes} {func_name}({params}) {{\n{body}\n}}\n")
});

// OpenCL allocates local memory in the kernel's scope, sized at compile time.
opencl_op!(AllocSharedOp, |op, ctx| {
    let name = op.get_result(ctx).name(ctx);
    let align = op.alignment(ctx).0;
    let size = op.size(ctx).0;
    format!("__local char __attribute__((aligned({align}))) {name}[{size}];\n")
});

opencl_op_with_out!(SliceSharedOp, |op, ctx| {
    let ptr_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    let block = op.block(ctx).name(ctx);
    let offset = op.offset(ctx).0;
    format!("({ptr_ty})(&{block}[{offset}])")
});

opencl_op!(DeclareVectorOp, |op, ctx| {
    let vector = op.vector_ty(ctx).get_type(ctx).deref(ctx);
    let vector = vector
        .downcast_ref::<VectorType>()
        .expect("a vector declaration should declare a vector type");
    if crate::opencl::ty::native_vector(ctx, vector).is_some() {
        return String::new();
    }
    let align = vector.align(ctx);
    let inner_ty = vector.inner.to_cpp(ctx);
    let vec = vector.vectorization;
    let fields = (0..vec).map(|i| format!("{inner_ty} i_{i};")).join(" ");
    format!("typedef struct __attribute__((aligned({align}))) {{ {fields} }} {inner_ty}_{vec};\n")
});

opencl_op!(IncludeOp, |_, _| String::new());
