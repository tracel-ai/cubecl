use core::{any::type_name, marker::PhantomData};

use derive_more::{Deref, DerefMut, From};
use derive_new::new;
use pliron::{
    attribute::AttrObj,
    builtin::ops::ConstantOp,
    debug_info::{self, set_operation_result_name},
    graph::walkers::uninterruptible::{
        immutable::{self},
        mutable,
    },
    irbuild::{
        dialect_conversion::apply_dialect_conversion,
        match_rewrite::{RewriterOrder, apply_match_rewrite},
    },
    op::OpInterfaceMarker,
};

use crate::{interfaces::SimplifyInterface, prelude::*};

#[derive(new, From, Clone, Debug, Default)]
pub struct DialectConversionPass<T: DialectConversion>(T);

impl<T: DialectConversion> Pass for DialectConversionPass<T> {
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
        res.ir_changed = apply_dialect_conversion(ctx, &mut self.0, op)?;
        Ok(res)
    }
}

#[derive(new, From, Clone, Debug, Default)]
pub struct MatchRewritePass<T: MatchRewrite>(pub T);

impl<T: MatchRewrite> Pass for MatchRewritePass<T> {
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
        res.ir_changed = apply_match_rewrite(ctx, &mut self.0, RewriterOrder::default(), op)?;
        Ok(res)
    }
}

#[derive(new, Clone, Copy, Default, Debug)]
pub struct CombinedPass<P1: Pass, P2: Pass> {
    pass_1: P1,
    pass_2: P2,
}

impl<P1: Pass, P2: Pass> Pass for CombinedPass<P1, P2> {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.ir_changed |= self.pass_1.run(op, ctx, analyses)?.ir_changed;
        res.ir_changed |= self.pass_2.run(op, ctx, analyses)?.ir_changed;
        Ok(res)
    }
}

pub type SimplifyOpsPass = MatchRewritePass<SimplifyOps>;

#[derive(Default, Clone, Copy)]
pub struct SimplifyOps;

impl MatchRewrite for SimplifyOps {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn SimplifyInterface>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let operand_attrs = const_operands(ctx, op);
        let simplify = op_cast::<dyn SimplifyInterface>(&*dyn_op).unwrap();
        if let Some(value) = simplify.check_fold(ctx, &operand_attrs) {
            rewriter.replace_operation_with_values(ctx, op, vec![value]);
        }
        Ok(())
    }
}

fn const_operands(ctx: &Context, op: Ptr<Operation>) -> Vec<Option<AttrObj>> {
    op.deref(ctx)
        .operands()
        .map(|opd| Some(opd.defining_op()?.as_op::<ConstantOp>(ctx)?.get_value(ctx)))
        .collect()
}

pub type VisitOpsCallback<T, State> = fn(&Context, &mut State, T);
pub type VisitOpsMutCallback<T, State> = fn(&mut Context, &mut State, T);

pub fn visit_all_ops_of_type<T: Op, State>(
    ctx: &Context,
    state: &mut State,
    root: Ptr<Operation>,
    callback: VisitOpsCallback<T, State>,
) {
    immutable::walk_op(
        ctx,
        &mut (state, callback),
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, (state, callback), node| {
            if let IRNode::Operation(op) = node
                && let Some(op) = op.as_op::<T>(ctx)
            {
                callback(ctx, state, op)
            }
        },
    );
}

pub fn visit_all_ops_of_type_mut<T: Op, State>(
    ctx: &mut Context,
    state: &mut State,
    root: Ptr<Operation>,
    callback: VisitOpsMutCallback<T, State>,
) {
    mutable::walk_op(
        ctx,
        &mut (state, callback),
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, (state, callback), node| {
            if let IRNode::Operation(op) = node
                && let Some(op) = op.as_op::<T>(ctx)
            {
                callback(ctx, state, op)
            }
        },
    );
}

pub fn visit_all_ops_with_interface<T: ?Sized + OpInterfaceMarker + 'static, State>(
    ctx: &Context,
    state: &mut State,
    root: Ptr<Operation>,
    callback: for<'a> fn(&Context, &mut State, &'a T),
) {
    immutable::walk_op(
        ctx,
        &mut (state, callback),
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, (state, callback), node| {
            if let IRNode::Operation(op) = node {
                let dyn_op = op.dyn_op(ctx);
                if let Some(op) = op_cast::<T>(&*dyn_op) {
                    callback(ctx, state, op)
                }
            }
        },
    );
}

pub trait RewriteOp<T: Op> {
    fn should_rewrite(&self, _ctx: &Context, _op: T) -> bool {
        true
    }

    fn rewrite(&mut self, ctx: &mut Context, rewriter: &mut MatchRewriter, op: T);
}

#[derive(new, Deref, DerefMut)]
pub struct MatchRewriteOp<T: Op, R: RewriteOp<T>> {
    #[deref]
    #[deref_mut]
    pub inner: R,
    _t: PhantomData<T>,
}

impl<T: Op, R: RewriteOp<T>> From<R> for MatchRewriteOp<T, R> {
    fn from(value: R) -> Self {
        Self::new(value)
    }
}

impl<T: Op, R: RewriteOp<T>> MatchRewrite for MatchRewriteOp<T, R> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.as_op::<T>(ctx)
            .is_some_and(|op| self.should_rewrite(ctx, op))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let op = op.as_op::<T>(ctx).unwrap();
        RewriteOp::rewrite(&mut self.inner, ctx, rewriter, op);
        Ok(())
    }
}

pub trait RewriterExt: Rewriter {
    fn replace_op_with(&mut self, ctx: &mut Context, op: Ptr<Operation>, new_op: Ptr<Operation>) {
        new_op.insert_before(ctx, op);
        transfer_result_names(ctx, op, &new_op.results(ctx));
        self.replace_operation(ctx, op, new_op);
    }
}
impl<R: Rewriter> RewriterExt for R {}

pub fn transfer_result_names(ctx: &Context, old_op: Ptr<Operation>, values: &[Value]) {
    for (idx, value) in values.iter().enumerate() {
        transfer_result_name(ctx, old_op, *value, idx);
    }
}

pub fn transfer_result_name(ctx: &Context, old_op: Ptr<Operation>, value: Value, idx: usize) {
    if let Some(new_op) = value.defining_op() {
        set_operation_result_name(
            ctx,
            new_op,
            idx,
            debug_info::get_operation_result_name(ctx, old_op, idx),
        );
    }
}
