//! The visibility analysis is a correctness oracle, not a hint.
//!
//! Downstream, a buffer stamped `Dead` gets no write tracking at all, so the
//! analysis has to fail wide: an access nobody can attribute must widen
//! visibility, never fall out of it. These tests pin the three ways an access
//! reaches a buffer — a traced pointer chain, an untraceable pointer whose
//! type still names its buffer, and inline asm that names nothing — and the
//! attributes the pass stamps from the answers.

use cubecl_ir::{
    AddressSpace, AddressType, Scope,
    attributes::{
        ATTR_BUFFER_BINDING, ATTR_BUFFER_IO, BufferBindingAttr, BufferIOAttr, FuncInterface,
    },
    dialect::{
        InputKind, InputSpec, MemoryClobbers, RegSpec,
        asm::InlineAsmOp,
        memory::{IndexOp, LoadOp, StoreOp},
        tma::{TmaLoadOp, TmaStoreOp},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::{
        PointerType,
        barrier::{BarrierLevel, BarrierType},
        scalar::IndexType,
    },
};
use cubecl_opt::{
    BufferVisibility, analyses::pointer_source::GlobalVisibility,
    passes::annotate_buffer_visibility::AnnotateGlobalVisibilityPass,
};
use pliron::{
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, Pass},
    r#type::TypeHandle,
    value::Value,
};

fn kernel() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 1, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

fn global(scope: &Scope, pos: usize) -> Value {
    let value_ty = IndexType::get(scope.ctx_mut()).to_handle();
    scope.global(pos, None, value_ty)
}

/// A pointer to element 0 of `buffer`, through the traced chain every
/// ordinary access takes.
fn element(scope: &Scope, buffer: Value) -> Value {
    let index = scope.const_usize(0);
    let op = IndexOp::new(scope.ctx_mut(), buffer, index, None);
    scope.register_with_result(&op)
}

fn load(scope: &Scope, ptr: Value) -> Value {
    let op = LoadOp::new(scope.ctx_mut(), ptr);
    scope.register_with_result(&op)
}

fn store(scope: &Scope, ptr: Value, value: Value) {
    let op = StoreOp::new(scope.ctx_mut(), ptr, value);
    scope.register(&op);
}

/// A value of type `ty` from inline asm that touches no memory, so that the op it feeds is the
/// only access the analysis sees.
fn opaque(scope: &Scope, ty: TypeHandle) -> Value {
    let asm = InlineAsmOp::new(
        scope.ctx_mut(),
        vec![ty],
        vec![RegSpec::Inferred],
        "opaque".into(),
        MemoryClobbers::Nomem,
        vec![],
        vec![],
    );
    scope.register(&asm);
    asm.results(scope.ctx())[0]
}

/// A pointer to shared memory, as TMA copies' other side is.
fn shared(scope: &Scope) -> Value {
    let ctx = scope.ctx_mut();
    let inner = IndexType::get(ctx);
    let ty = PointerType::get(ctx, inner.into(), AddressSpace::Shared).to_handle();
    opaque(scope, ty)
}

/// A TMA load of a 2D box through `tensor_map` into shared memory.
fn tma_load(scope: &Scope, tensor_map: Value) {
    let barrier = {
        let ctx = scope.ctx_mut();
        let barrier = BarrierType::get(ctx, BarrierLevel::Cube);
        let ty = PointerType::get(ctx, barrier.into(), AddressSpace::Shared).to_handle();
        opaque(scope, ty)
    };
    let destination = shared(scope);
    let index = scope.const_usize(0);
    let op = TmaLoadOp::new(
        scope.ctx_mut(),
        barrier,
        tensor_map,
        destination,
        vec![index, index],
    );
    scope.register(&op);
}

/// A TMA store of a 2D box from shared memory through `tensor_map`.
fn tma_store(scope: &Scope, tensor_map: Value) {
    let source = shared(scope);
    let index = scope.const_usize(0);
    let op = TmaStoreOp::new(scope.ctx_mut(), source, tensor_map, vec![index, index]);
    scope.register(&op);
}

/// Finish the kernel and compute the analysis.
fn visibility(scope: Scope) -> std::collections::HashMap<usize, BufferVisibility> {
    let (module_op, _, ctx) = finish(scope);
    let mut analyses = AnalysisManager::default();
    let global = analyses
        .get_analysis::<GlobalVisibility>(module_op, &ctx)
        .expect("the analysis computes");
    global.visibility.iter().map(|(k, v)| (*k, *v)).collect()
}

fn finish(scope: Scope) -> (Ptr<Operation>, FuncOp, Context) {
    let module_op = scope.state().module.get_operation();
    let entry_func = scope.state().entry_func;
    let ctx = scope.into_context().expect("the scope owns its context");
    (module_op, entry_func, ctx)
}

#[test]
fn loads_and_stores_mark_exactly_the_buffers_they_touch() {
    let scope = kernel();
    let read = global(&scope, 0);
    let written = global(&scope, 1);
    let untouched = global(&scope, 2);
    let _ = untouched;

    let value = load(&scope, element(&scope, read));
    store(&scope, element(&scope, written), value);

    let visibility = visibility(scope);
    assert_eq!(
        visibility[&0],
        BufferVisibility {
            readable: true,
            writable: false
        }
    );
    assert_eq!(
        visibility[&1],
        BufferVisibility {
            readable: false,
            writable: true
        }
    );
    assert_eq!(
        visibility[&2],
        BufferVisibility {
            readable: false,
            writable: false
        }
    );
}

/// The failure direction the hardening exists for: a pointer the source
/// analysis cannot follow used to drop its access on the floor, leaving the
/// buffer stamped `Dead`. The pointer's own type carries its address space —
/// and for globals, the binding index — so the access lands on the one buffer
/// the type names.
#[test]
fn an_untraceable_global_pointer_is_attributed_by_its_type() {
    let scope = kernel();
    let _reachable = global(&scope, 0);
    let _mystery = global(&scope, 1);

    // Inline asm hands back a pointer into buffer 1: no aliasing chain to
    // follow, only the type. `nomem` keeps the asm's own effects out of the
    // picture, so what is tested is the load through the untraced result.
    let ptr = {
        let ctx = scope.ctx_mut();
        let inner = IndexType::get(ctx);
        let ptr_ty = PointerType::get(ctx, inner.into(), AddressSpace::Global(1));
        InlineAsmOp::new(
            ctx,
            vec![ptr_ty.to_handle()],
            vec![RegSpec::Inferred],
            "mystery_pointer".into(),
            MemoryClobbers::Nomem,
            vec![],
            vec![],
        )
    };
    scope.register(&ptr);
    let ptr = ptr.results(scope.ctx())[0];
    load(&scope, ptr);

    let visibility = visibility(scope);
    assert_eq!(
        visibility[&1],
        BufferVisibility {
            readable: true,
            writable: false
        },
        "the untraced access lands on the buffer its type names"
    );
    assert_eq!(
        visibility[&0],
        BufferVisibility {
            readable: false,
            writable: false
        },
        "and on no other"
    );
}

/// Inline asm that names no pointer can touch any buffer the kernel holds,
/// so every buffer widens to what the asm could have done.
#[test]
fn inline_asm_reaches_every_buffer() {
    let scope = kernel();
    let _a = global(&scope, 0);
    let _b = global(&scope, 1);

    let asm = InlineAsmOp::new(
        scope.ctx_mut(),
        vec![],
        vec![],
        "who_knows".into(),
        MemoryClobbers::ReadWrite,
        vec![],
        vec![],
    );
    scope.register(&asm);

    let visibility = visibility(scope);
    for pos in [0, 1] {
        assert_eq!(
            visibility[&pos],
            BufferVisibility {
                readable: true,
                writable: true
            },
            "asm with unknown effects widens buffer {pos} fully"
        );
    }
}

/// Readonly asm reads anywhere but writes nowhere, and the distinction
/// survives into the analysis.
#[test]
fn readonly_inline_asm_marks_reads_only() {
    let scope = kernel();
    let _a = global(&scope, 0);

    let asm = InlineAsmOp::new(
        scope.ctx_mut(),
        vec![],
        vec![],
        "observer".into(),
        MemoryClobbers::Readonly,
        vec![],
        vec![],
    );
    scope.register(&asm);

    let visibility = visibility(scope);
    assert_eq!(
        visibility[&0],
        BufferVisibility {
            readable: true,
            writable: false
        }
    );
}

/// A tensor map is an argument of its own type, not a pointer, so no pointer chain reaches it:
/// what the kernel does with the tensor behind it is what the TMA ops taking it do. A map only
/// loaded through is read and never written, which is what keeps a launch that fails before it
/// runs from leaving the tensor it reads carrying the failure.
#[test]
fn tensor_maps_are_read_by_loads_and_written_by_stores() {
    let scope = kernel();
    let loaded = scope.tensor_map(0, 0);
    let stored = scope.tensor_map(1, 1);
    let _unused = scope.tensor_map(2, 2);
    let both = scope.tensor_map(3, 3);

    tma_load(&scope, loaded);
    tma_store(&scope, stored);
    tma_load(&scope, both);
    tma_store(&scope, both);

    let visibility = visibility(scope);
    let expect = |readable, writable| Some(BufferVisibility { readable, writable });
    let got = |pos| visibility.get(&pos).copied();
    assert_eq!(got(0), expect(true, false), "a map only loaded is read");
    assert_eq!(got(1), expect(false, true), "a map only stored is written");
    assert_eq!(got(2), expect(false, false), "an unused map is dead");
    assert_eq!(
        got(3),
        expect(true, true),
        "a map loaded and stored is both"
    );
}

/// What the analysis cannot follow it must not narrow: a tensor map handed to anything but a
/// TMA copy, here inline asm that claims to touch no memory itself, may still be read or
/// written through, so it is both.
#[test]
fn a_tensor_map_used_by_anything_else_is_read_and_written() {
    let scope = kernel();
    let map = scope.tensor_map(0, 0);
    let asm = InlineAsmOp::new(
        scope.ctx_mut(),
        vec![],
        vec![],
        "uses_the_map".into(),
        MemoryClobbers::Nomem,
        vec![map],
        vec![InputSpec {
            kind: InputKind::In,
            class: RegSpec::Inferred,
        }],
    );
    scope.register(&asm);
    tma_load(&scope, map);

    let visibility = visibility(scope);
    assert_eq!(
        visibility.get(&0).copied(),
        Some(BufferVisibility {
            readable: true,
            writable: true
        })
    );
}

/// The pass end to end: what the analysis concludes is what the entry
/// function's arguments get stamped with.
#[test]
fn the_pass_stamps_the_attributes_the_analysis_concluded() {
    let scope = kernel();
    let read = global(&scope, 0);
    let written = global(&scope, 1);
    let _untouched = global(&scope, 2);
    let loaded = scope.tensor_map(3, 3);

    let value = load(&scope, element(&scope, read));
    store(&scope, element(&scope, written), value);
    tma_load(&scope, loaded);

    let (module_op, entry_func, mut ctx) = finish(scope);
    let mut analyses = AnalysisManager::default();
    AnnotateGlobalVisibilityPass
        .run(module_op, &mut ctx, &mut analyses)
        .expect("the pass runs");

    let num_args = entry_func
        .get_entry_block(&ctx)
        .deref(&ctx)
        .get_num_arguments();
    let mut stamped = std::collections::HashMap::new();
    for arg in 0..num_args {
        let binding = entry_func
            .get_arg_attr::<BufferBindingAttr>(&ctx, arg, &ATTR_BUFFER_BINDING)
            .map(|it| *it);
        let io = entry_func
            .get_arg_attr::<BufferIOAttr>(&ctx, arg, &ATTR_BUFFER_IO)
            .map(|it| *it);
        if let (Some(binding), Some(io)) = (binding, io) {
            stamped.insert(binding.buffer_pos, io);
        }
    }

    assert_eq!(stamped[&0], BufferIOAttr::ReadOnly);
    assert_eq!(stamped[&1], BufferIOAttr::WriteOnly);
    assert_eq!(stamped[&2], BufferIOAttr::Dead);
    assert_eq!(
        stamped.get(&3),
        Some(&BufferIOAttr::ReadOnly),
        "a tensor map gets its stamp too, rather than reading as both"
    );
}
