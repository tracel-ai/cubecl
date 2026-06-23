use core::{any::type_name, cell::RefCell};

use derive_more::From;
use derive_new::new;
use pliron::{
    attribute::AttrObj,
    builtin::ops::ConstantOp,
    irbuild::{dialect_conversion::apply_dialect_conversion, match_rewrite::apply_match_rewrite},
};

use crate::{interfaces::SimplifyInterface, prelude::*};

#[derive(new, From, Clone, Debug)]
pub struct DialectConversionPass<T: DialectConversion>(pub RefCell<T>);

impl<T: DialectConversion + Default> Default for DialectConversionPass<T> {
    fn default() -> Self {
        Self(RefCell::new(Default::default()))
    }
}

impl<T: DialectConversion> Pass for DialectConversionPass<T> {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let mut convert = self.0.borrow_mut();
        res.ir_changed = apply_dialect_conversion(ctx, &mut *convert, op)?;
        Ok(res)
    }
}

#[derive(new, From, Clone, Debug, Default)]
pub struct MatchRewritePass<T: MatchRewrite>(pub T);

impl<T: MatchRewrite + Clone> Pass for MatchRewritePass<T> {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.ir_changed = apply_match_rewrite(ctx, self.0.clone(), op)?;
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
        &self,
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
