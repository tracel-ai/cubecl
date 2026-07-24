use alloc::{boxed::Box, vec, vec::Vec};
use cubecl_ir::{
    VectorSize,
    attributes::IndexAttr,
    dialect::{
        base::OperationPtrExt,
        general::CopyOp,
        math::{IAddOp, IMulOp},
        matrix,
        memory::{DeclareVariableOp, IndexOp},
        vector::{VectorExtractOp, VectorInsertOp},
    },
    interfaces::{MaybeVectorizedType, TriviallyUnrollable, TypedExt},
    prelude::*,
    try_cast_op,
    types::{ArrayType, AtomicType, PointerType, RuntimeArrayType, VectorType},
    verify_op_succ, verify_ty_succ,
};
use hashbrown::HashMap;
use pliron::{
    builtin::{
        ops::{ConstantOp, FuncOp},
        types::FunctionType,
    },
    graph::walkers::{WALKCONFIG_PREORDER_FORWARD, uninterruptible::mutable::walk_op},
    printable::Printable,
};

type Mappings = HashMap<Value, Vec<Value>>;

#[derive(Debug, new)]
pub struct UnrollPass {
    max_vector_size: VectorSize,
}

#[type_interface]
pub trait UnrollableType: MaybeVectorizedType {
    verify_ty_succ!();
    fn with_vector_size(&self, ctx: &Context, vectorization: usize) -> TypeHandle;
}

#[type_interface_impl]
impl UnrollableType for VectorType {
    fn with_vector_size(&self, ctx: &Context, vectorization: usize) -> TypeHandle {
        VectorType::get(ctx, self.inner, vectorization).into()
    }
}

#[type_interface_impl]
impl UnrollableType for AtomicType {
    fn with_vector_size(&self, ctx: &Context, vectorization: usize) -> TypeHandle {
        let inner = self.inner.deref(ctx);
        let unrollable = type_cast::<dyn UnrollableType>(&*inner).expect("Should be implemented");
        let new_inner = unrollable.with_vector_size(ctx, vectorization);
        AtomicType::get(ctx, new_inner).into()
    }
}

#[type_interface_impl]
impl UnrollableType for PointerType {
    fn with_vector_size(&self, ctx: &Context, vectorization: usize) -> TypeHandle {
        let inner = self.inner.deref(ctx);
        let unrollable = type_cast::<dyn UnrollableType>(&*inner).expect("Should be implemented");
        let new_inner = unrollable.with_vector_size(ctx, vectorization);
        PointerType::get(ctx, new_inner, self.address_space).into()
    }
}

#[type_interface_impl]
impl UnrollableType for ArrayType {
    fn with_vector_size(&self, ctx: &Context, new_vec: usize) -> TypeHandle {
        let current_vec = self.vector_size(ctx);
        let inner = self.inner.deref(ctx);
        let unrollable = type_cast::<dyn UnrollableType>(&*inner).expect("Should be implemented");
        let new_inner = unrollable.with_vector_size(ctx, new_vec);
        ArrayType::get(ctx, new_inner, self.length * current_vec / new_vec).into()
    }
}

#[type_interface_impl]
impl UnrollableType for RuntimeArrayType {
    fn with_vector_size(&self, ctx: &Context, new_vec: usize) -> TypeHandle {
        let inner = self.inner.deref(ctx);
        let unrollable = type_cast::<dyn UnrollableType>(&*inner).expect("Should be implemented");
        let new_inner = unrollable.with_vector_size(ctx, new_vec);
        RuntimeArrayType::get(ctx, new_inner).into()
    }
}

#[op_interface]
pub trait CustomUnrollOp {
    verify_op_succ!();
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState);
}

#[op_interface_impl]
impl CustomUnrollOp for DeclareVariableOp {
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState) {
        let value_ty = self.value_ty(ctx).get_type(ctx);
        let current_vec = value_ty.try_get_vector_size(ctx).unwrap_or(1);
        if current_vec <= state.max_vector_size {
            return;
        }

        state.result.ir_changed |= IRStatus::Changed;
        let result = self.get_result(ctx);
        let addr_space = *self.addr_space(ctx);
        // Align isn't handled properly, but targets that unroll ignore this anyways
        let align = *self.alignment(ctx);
        let new_value_ty = unroll_ty(ctx, value_ty, state.max_vector_size);
        let new_ptr_ty = PointerType::get(ctx, new_value_ty, addr_space.0);

        // Array doesn't change size, so no need to duplicate the declaration
        if new_value_ty.size(ctx) == value_ty.size(ctx) {
            self.set_value_ty(ctx, new_value_ty);
            self.get_result(ctx).set_type(ctx, new_ptr_ty.into());
        } else {
            let factor = current_vec / state.max_vector_size;
            let mut results = vec![];
            for _ in 0..factor {
                let init = self.initializer(ctx).map(|init| init.clone());
                let new_op = DeclareVariableOp::new(ctx, new_value_ty, addr_space, align, init);
                new_op
                    .get_operation()
                    .insert_before(ctx, self.get_operation());
                results.push(new_op.get_result(ctx));
            }
            state.mappings.insert(result, results);
            state.to_erase.push(self.get_operation());
        }
    }
}

#[op_interface_impl]
impl CustomUnrollOp for IndexOp {
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState) {
        let base = self.base(ctx);
        let checked = self.checked(ctx);
        let current_vec = try_get_vec(ctx, self.get_result(ctx));
        if current_vec > state.max_vector_size {
            state.result.ir_changed |= IRStatus::Changed;
            let unroll_factor = current_vec / state.max_vector_size;
            let unroll_const = const_usize(ctx, self, unroll_factor);

            let mul = IMulOp::new(ctx, self.index(ctx), unroll_const);
            mul.get_operation().insert_before(ctx, self.get_operation());
            let start_idx = mul.get_result(ctx);

            let new_results = (0..unroll_factor)
                .map(|i| {
                    let i = const_usize(ctx, self, i);
                    let add = IAddOp::new(ctx, start_idx, i);
                    add.get_operation().insert_before(ctx, self.get_operation());
                    let idx = add.get_result(ctx);

                    let op = IndexOp::maybe_checked(ctx, base, idx, checked);
                    op.get_operation().insert_before(ctx, self.get_operation());
                    op.get_result(ctx)
                })
                .collect();

            state.mappings.insert(self.get_result(ctx), new_results);
            state.to_erase.push(self.get_operation());
        }
    }
}

fn const_usize(ctx: &mut Context, anchor: &dyn Op, value: usize) -> Value {
    let op = ConstantOp::new(ctx, Box::new(IndexAttr::new(value)));
    op.get_operation()
        .insert_before(ctx, anchor.get_operation());
    op.get_result(ctx)
}

#[op_interface_impl]
impl CustomUnrollOp for VectorExtractOp {
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState) {
        let vector = self.vector(ctx);
        let current_vec = vector.vector_size(ctx);
        if current_vec > state.max_vector_size {
            state.result.ir_changed |= IRStatus::Changed;
            let index = self.index(ctx).0;

            let unroll_idx = index / state.max_vector_size;
            let sub_idx = index % state.max_vector_size;

            let new_vector = state.mappings.get(&vector).expect("Should exist")[unroll_idx];
            vector.replace_use_with(ctx, self.vector_as_use(ctx), &new_vector);
            self.set_index(ctx, sub_idx);
        }
    }
}

#[op_interface_impl]
impl CustomUnrollOp for VectorInsertOp {
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState) {
        let vector = self.vector(ctx);
        let value = self.value(ctx);
        let current_vec = vector.vector_size(ctx);
        if current_vec > state.max_vector_size {
            state.result.ir_changed |= IRStatus::Changed;
            let index = self.index(ctx).0;

            let unroll_idx = index / state.max_vector_size;
            let sub_idx = index % state.max_vector_size;

            let vectors = state.mappings.get(&vector).expect("Should exist");

            let new_results = vectors.iter().enumerate().map(|(i, vector)| {
                let op = if i == unroll_idx {
                    VectorInsertOp::new(ctx, *vector, value, sub_idx).get_operation()
                } else {
                    CopyOp::new(ctx, *vector).get_operation()
                };
                op.insert_before(ctx, self.get_operation());
                op.deref(ctx).get_result(0)
            });
            let new_results = new_results.collect();
            state.mappings.insert(self.get_result(ctx), new_results);
            state.to_erase.push(self.get_operation());
        }
    }
}

#[op_interface_impl]
impl CustomUnrollOp for matrix::LoadOp {
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState) {
        let source = self.source(ctx);
        if source.vector_size(ctx) > state.max_vector_size {
            state.result.ir_changed |= IRStatus::Changed;
            let new_source = state.mappings.get(&source).expect("should exist")[0];
            source.replace_use_with(ctx, self.source_as_use(ctx), &new_source);
        }
    }
}

#[op_interface_impl]
impl CustomUnrollOp for matrix::StoreOp {
    fn unroll(&self, ctx: &mut Context, state: &mut UnrollState) {
        let dest = self.destination(ctx);
        if dest.vector_size(ctx) > state.max_vector_size {
            state.result.ir_changed |= IRStatus::Changed;
            let new_dest = state.mappings.get(&dest).expect("should exist")[0];
            dest.replace_use_with(ctx, self.destination_as_use(ctx), &new_dest);
        }
    }
}

pub struct UnrollState {
    mappings: Mappings,
    to_erase: Vec<Ptr<Operation>>,
    max_vector_size: VectorSize,
    result: PassResult,
    rewriter: PassRewriter,
}

#[pass_name]
impl Pass for UnrollPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        self.unroll_func(ctx, op);

        let mut state = UnrollState {
            mappings: Default::default(),
            to_erase: Default::default(),
            max_vector_size: self.max_vector_size,
            result: Default::default(),
            rewriter: PassRewriter::default(),
        };

        walk_op(
            ctx,
            &mut state,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, state, node| {
                if let IRNode::Operation(op) = node {
                    let unroll_opds = op
                        .operands(ctx)
                        .iter()
                        .any(|it| should_unroll(ctx, it, state.max_vector_size));
                    let unroll_res = op
                        .results(ctx)
                        .iter()
                        .any(|it| should_unroll(ctx, it, state.max_vector_size));
                    let dyn_op = op.dyn_op(ctx);

                    if let Some(custom) = op_cast::<dyn CustomUnrollOp>(&*dyn_op) {
                        custom.unroll(ctx, state);
                    } else if unroll_opds || unroll_res {
                        state.result.ir_changed |= IRStatus::Changed;
                        unroll_default(ctx, state, op);
                    }
                }
            },
        );

        while !state.to_erase.is_empty() {
            // Pop the next op that no longer has uses. This ensures we always start at the end of
            // the def-use chain
            let next = state
                .to_erase
                .iter()
                .position(|it| !it.deref(ctx).has_use())
                .expect("Erased ops should only have uses in other erased ops");
            let op = state.to_erase.remove(next);
            state.rewriter.erase_operation(ctx, op);
        }

        Ok(state.result)
    }
}

impl UnrollPass {
    fn unroll_func(&self, ctx: &mut Context, op: Ptr<Operation>) {
        let func = op.as_op::<FuncOp>(ctx).expect("Should be func");
        let entry_block = func.get_entry_block(ctx);
        let func_ty = func.get_attr_func_type(ctx).unwrap().get_type(ctx);
        let func_ty = func_ty.deref(ctx);
        let func_ty = func_ty.downcast_ref::<FunctionType>().unwrap();

        let mut new_func_inputs = vec![];

        for (i, arg) in func_ty.arg_types().into_iter().enumerate() {
            if should_unroll(ctx, arg, self.max_vector_size) {
                let new_ty = unroll_ty(ctx, arg, self.max_vector_size);
                new_func_inputs.push(new_ty);
                let block_arg = entry_block.deref(ctx).get_argument(i);
                block_arg.set_type(ctx, new_ty);
            } else {
                new_func_inputs.push(arg);
            }
        }

        let new_func_ty = FunctionType::get(ctx, new_func_inputs, func_ty.res_types()).to_handle();
        func.set_attr_func_type(ctx, new_func_ty.into());
    }
}

fn unroll_default(ctx: &mut Context, state: &mut UnrollState, op: Ptr<Operation>) {
    let values = op.operands(ctx).into_iter().chain(op.results(ctx));
    let current_vec = values.map(|it| try_get_vec(ctx, it)).max().unwrap();
    let factor = current_vec / state.max_vector_size;
    let dyn_op = op.dyn_op(ctx);
    let rematerialize = try_cast_op!(dyn_op, ctx, dyn TriviallyUnrollable);
    let new_out_ty = op
        .results(ctx)
        .into_iter()
        .map(|it| unroll_ty(ctx, it, state.max_vector_size))
        .collect::<Vec<_>>();
    let mut new_results = vec![];

    for unroll_idx in 0..factor {
        let opds = op.operands(ctx).into_iter().map(|opd| {
            if should_unroll(ctx, opd, state.max_vector_size) {
                if !state.mappings.contains_key(&opd) {
                    std::println!("opd: {}, mappings: {:?}", opd.disp(ctx), state.mappings);
                }
                state.mappings.get(&opd).expect("Should have mapping")[unroll_idx]
            } else {
                opd
            }
        });
        let attrs = op.deref(ctx).attributes.clone();
        let new_op = rematerialize.materialize(ctx, new_out_ty.clone(), opds.collect(), attrs);
        new_results.extend(new_op.deref(ctx).results());
        new_op.insert_before(ctx, op);
    }

    if !new_results.is_empty() {
        state
            .mappings
            .insert(op.deref(ctx).get_result(0), new_results);
    }
    state.to_erase.push(op);
}

fn should_unroll(ctx: &Context, value: impl Typed, max_vector_size: usize) -> bool {
    let ty = value.get_type(ctx).deref(ctx);
    if !type_impls::<dyn UnrollableType>(&*ty) {
        return false;
    }
    let Some(vector_size) = value.try_get_vector_size(ctx) else {
        return false;
    };
    vector_size > max_vector_size
}

fn try_get_vec(ctx: &Context, value: impl Typed) -> usize {
    value.try_get_vector_size(ctx).unwrap_or(1)
}

fn unroll_ty(ctx: &Context, ty: impl Typed, vectorization: usize) -> TypeHandle {
    let ty = ty.get_type(ctx).deref(ctx);
    type_cast::<dyn UnrollableType>(&*ty)
        .expect("Should be unrollable")
        .with_vector_size(ctx, vectorization)
}
