use cubecl_core::{
    ir::{
        AddressType, Scope,
        dialect::{
            general::{CastOp, CopyOp},
            memory::LoadOp,
            vector::CompositeConstructOp,
        },
        interfaces::TypedExt,
        prelude::*,
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::{
            VectorType,
            scalar::{Float4E2M1x2Type, Float16Type, Float32Type},
        },
    },
    post_processing::unroll::UnrollPass,
};
use pliron::{basic_block::BasicBlock, linked_list::ContainsLinkedList};

fn cast_widths(ctx: &Context, entry: Ptr<BasicBlock>) -> Vec<(usize, usize)> {
    entry
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| op.as_op::<CastOp>(ctx))
        .map(|cast| {
            (
                cast.input(ctx).vector_size(ctx),
                cast.get_result(ctx).vector_size(ctx),
            )
        })
        .collect()
}

fn run_cast_pass(
    types: impl FnOnce(&Context) -> (TypeHandle, TypeHandle),
    add_consumer: bool,
) -> (Context, Ptr<BasicBlock>, Value) {
    let scope = Scope::root(KernelSettings::new(
        Dim3::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let (input_ty, output_ty) = types(scope.ctx());
    let input_ptr = scope.create_local_mut(input_ty, None);
    let load = LoadOp::new(scope.ctx_mut(), input_ptr);
    let input = scope.register_with_result(&load);
    let cast = CastOp::new(scope.ctx_mut(), output_ty, input);
    let result = cast.get_result(scope.ctx());
    scope.register(&cast);
    if add_consumer {
        scope.register(&CopyOp::new(scope.ctx_mut(), result));
    }

    let func = scope.state().entry_func;
    let mut ctx = scope
        .into_context()
        .expect("the root scope owns its context");
    UnrollPass::new(4)
        .run(
            func.get_operation(),
            &mut ctx,
            &mut AnalysisManager::default(),
        )
        .expect("unroll pass should run");
    let entry = func.get_entry_block(&ctx);
    (ctx, entry, input)
}

/// Reproduces https://github.com/tracel-ai/cubecl/issues/1503.
#[test]
fn cast_from_e2m1x2_splits_packed_input_with_output() {
    let (ctx, entry, _) = run_cast_pass(
        |ctx| {
            let packed = Float4E2M1x2Type::get(ctx).to_handle();
            let f16 = Float16Type::get(ctx).to_handle();
            (
                VectorType::get(ctx, packed, 4).to_handle(),
                VectorType::get(ctx, f16, 8).to_handle(),
            )
        },
        false,
    );

    let casts = entry
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| op.as_op::<CastOp>(&ctx))
        .collect::<Vec<_>>();
    assert_eq!(casts.len(), 2, "vec8 output should become two vec4 casts");

    let first_input = casts[0].input(&ctx);
    let second_input = casts[1].input(&ctx);
    assert_eq!(first_input.vector_size(&ctx), 2);
    assert_eq!(second_input.vector_size(&ctx), 2);
    assert_ne!(
        first_input, second_input,
        "casts must use distinct input chunks"
    );
    assert_eq!(casts[0].get_result(&ctx).vector_size(&ctx), 4);
    assert_eq!(casts[1].get_result(&ctx).vector_size(&ctx), 4);
}

#[test]
fn cast_broadcasts_scalar_to_each_output_piece() {
    let (ctx, entry, input) = run_cast_pass(
        |ctx| {
            let f32 = Float32Type::get(ctx).to_handle();
            let f16 = Float16Type::get(ctx).to_handle();
            (f32, VectorType::get(ctx, f16, 8).to_handle())
        },
        false,
    );

    let casts = entry
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| op.as_op::<CastOp>(&ctx))
        .collect::<Vec<_>>();
    assert_eq!(casts.len(), 2);
    for cast in casts {
        assert_eq!(cast.input(&ctx), input);
        assert_eq!(cast.get_result(&ctx).vector_size(&ctx), 4);
    }
}

#[test]
fn cast_with_wide_input_reconstructs_fitting_packed_output() {
    let (ctx, entry, _) = run_cast_pass(
        |ctx| {
            let f16 = Float16Type::get(ctx).to_handle();
            let packed = Float4E2M1x2Type::get(ctx).to_handle();
            (
                VectorType::get(ctx, f16, 8).to_handle(),
                VectorType::get(ctx, packed, 4).to_handle(),
            )
        },
        true,
    );

    assert_eq!(cast_widths(&ctx, entry), vec![(4, 2), (4, 2)]);
    let joined = entry
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| op.as_op::<CompositeConstructOp>(&ctx))
        .next()
        .expect("the fitting output should be reconstructed");
    let consumer = entry
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|op| op.as_op::<CopyOp>(&ctx))
        .expect("the cast should retain its consumer");
    assert_eq!(consumer.value(&ctx), joined.get_result(&ctx));
    assert_eq!(joined.get_result(&ctx).vector_size(&ctx), 4);
}
