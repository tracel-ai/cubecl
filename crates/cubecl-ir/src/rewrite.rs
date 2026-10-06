use core::{
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
    ops::{Deref, Range},
};

use cubecl_macros_internal::NamedRewrite;
use derive_more::{Deref, DerefMut, From};
use derive_new::new;
use pliron::{
    attribute::{AttrObj, Attribute},
    builtin::{
        given_names::{get_operation_result_name, set_operation_result_name},
        ops::ConstantOp,
    },
    graph::walkers::{
        WalkConfig,
        uninterruptible::{
            immutable::{self},
            mutable,
        },
    },
    irbuild::{
        dialect_conversion::apply_dialect_conversion,
        listener::RecorderEvent,
        match_rewrite::{RewriterOrder, apply_match_rewrite},
    },
    linked_list::ContainsLinkedList,
    location::Location,
    op::{OpInterfaceMarker, OpObj},
    value::Use,
    verify_err_noloc,
};

use crate::{
    dialect::BlockPtrExt,
    interfaces::{CanonicalizeInterface, SimplifyInterface},
    prelude::*,
};

/// A preset config when order doesn't matter
pub const WALKCONFIG_ANY: WalkConfig = WALKCONFIG_PREORDER_FORWARD;

pub trait NamedRewrite {
    fn name(&self) -> &str;
}

#[derive(new, From, Clone, Debug, Default)]
pub struct DialectConversionPass<T: DialectConversion>(T);

impl<T: DialectConversion + NamedRewrite> Pass for DialectConversionPass<T> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.ir_changed = apply_dialect_conversion(ctx, &mut KeepLocation(&mut self.0), op)?;
        Ok(res)
    }
}

#[derive(new, From, Clone, Debug, Default)]
pub struct MatchRewritePass<T: MatchRewrite>(pub T);

impl<T: MatchRewrite + NamedRewrite> Pass for MatchRewritePass<T> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.ir_changed = apply_match_rewrite(
            ctx,
            &mut KeepLocation(&mut self.0),
            RewriterOrder::default(),
            op,
        )?;
        Ok(res)
    }
}

/// A rewrite that gives each op it inserts without a location the location of the op it
/// rewrites. [`DialectConversionPass`] and [`MatchRewritePass`] use it for every rewrite.
pub struct KeepLocation<'a, T>(pub &'a mut T);

impl<T: DialectConversion> DialectConversion for KeepLocation<'_, T> {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        self.0.can_convert_op(ctx, op)
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        self.0.can_convert_type(ctx, ty)
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        self.0.convert_type(ctx, ty)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        keep_location(ctx, rewriter, op, |ctx, rewriter| {
            self.0.rewrite(ctx, rewriter, op, operands_info)
        })
    }
}

impl<T: MatchRewrite> MatchRewrite for KeepLocation<'_, T> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        self.0.r#match(ctx, op)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        keep_location(ctx, rewriter, op, |ctx, rewriter| {
            self.0.rewrite(ctx, rewriter, op)
        })
    }
}

/// Runs `rewrite` on `op`, then gives the location of `op` to each op that `rewrite` inserted
/// without one.
fn keep_location(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<Recorder>,
    op: Ptr<Operation>,
    rewrite: impl FnOnce(&mut Context, &mut IRRewriter<Recorder>) -> Result<()>,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if loc.is_unknown() {
        return rewrite(ctx, rewriter);
    }
    let first = rewriter.get_listener().events.len();
    rewrite(ctx, rewriter)?;

    let events = &rewriter.get_listener().events[first..];
    // An op that the rewrite inserted and then erased is not live.
    let erased = events
        .iter()
        .filter_map(|event| match event {
            RecorderEvent::ErasedOperation(op) => Some(*op),
            _ => None,
        })
        .collect::<Vec<_>>();
    for event in events {
        if let RecorderEvent::InsertedOperation(new_op) = event
            && !erased.contains(new_op)
            && new_op.deref(ctx).loc().is_unknown()
        {
            new_op.deref_mut(ctx).set_loc(loc.clone());
        }
    }
    Ok(())
}

/// Gives each op without a location the location of the op before it in its block. The first op
/// of a block takes the location of the op that holds the block. Run it last before export, for
/// the ops that passes outside [`KeepLocation`] insert.
#[derive(Clone, Copy, Debug, Default)]
pub struct InheritLocationPass;

#[pass_name]
impl Pass for InheritLocationPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let loc = op.deref(ctx).loc();
        let mut res = PassResult::default();
        if inherit_locations(ctx, op, &loc) {
            res.ir_changed = IRStatus::Changed;
        }
        Ok(res)
    }
}

/// Gives the ops nested in `op` a location, where `loc` is the location of `op`. Returns whether
/// an op changed.
fn inherit_locations(ctx: &Context, op: Ptr<Operation>, loc: &Location) -> bool {
    let mut changed = false;
    // The iterators borrow each node only to step to the next one, so a child op can change.
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            let mut previous = loc.clone();
            for child in block.deref(ctx).iter(ctx) {
                let child_loc = child.deref(ctx).loc();
                if child_loc.is_unknown() {
                    if !previous.is_unknown() {
                        child.deref_mut(ctx).set_loc(previous.clone());
                        changed = true;
                    }
                } else {
                    previous = child_loc;
                }
                changed |= inherit_locations(ctx, child, &previous);
            }
        }
    }
    changed
}

#[derive(new, Clone, Copy, Default, Debug)]
pub struct CombinedPass<P1: Pass, P2: Pass> {
    pass_1: P1,
    pass_2: P2,
}

#[pass_name]
impl<P1: Pass, P2: Pass> Pass for CombinedPass<P1, P2> {
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

#[derive(Default, Clone, Copy, NamedRewrite)]
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

pub fn const_operand<T: Attribute>(ctx: &Context, op: Ptr<Operation>, idx: usize) -> Option<T> {
    Some(*const_operands(ctx, op).remove(idx)?.downcast().ok()?)
}

pub fn const_operands(ctx: &Context, op: Ptr<Operation>) -> Vec<Option<AttrObj>> {
    op.deref(ctx)
        .operands()
        .map(|opd| {
            Some(
                opd.defining_op()?
                    .as_op::<ConstantOp>(ctx)?
                    .get_attr_builtin_constant_value(ctx)?
                    .clone(),
            )
        })
        .collect()
}

pub type CanonicalizePass = MatchRewritePass<Canonicalize>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct Canonicalize;

impl MatchRewrite for Canonicalize {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn CanonicalizeInterface>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let canonicalize = op_cast::<dyn CanonicalizeInterface>(&*dyn_op).unwrap();
        canonicalize.canonicalize(ctx, rewriter)?;
        Ok(())
    }
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
        &WALKCONFIG_ANY,
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

pub fn visit_all_values<State>(
    ctx: &Context,
    state: &mut State,
    root: Ptr<Operation>,
    callback: for<'a> fn(&Context, &mut State, Value),
) {
    immutable::walk_op(
        ctx,
        &mut (state, callback),
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, (state, callback), node| match node {
            IRNode::Operation(ptr) => {
                for res in ptr.results(ctx) {
                    callback(ctx, state, res)
                }
            }
            IRNode::BasicBlock(ptr) => {
                for res in ptr.arguments(ctx) {
                    callback(ctx, state, res)
                }
            }
            IRNode::Region(_) => {}
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
    fn append_op_with_result(&mut self, ctx: &mut Context, op: &impl OneResultInterface) -> Value {
        self.append_op(ctx, op);
        op.get_result(ctx)
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
            get_operation_result_name(ctx, old_op, idx),
        );
    }
}

pub struct TraitOp<T: OpInterfaceMarker + ?Sized> {
    obj: OpObj,
    _marker: PhantomData<T>,
}

impl<T: OpInterfaceMarker + ?Sized> TraitOp<T> {
    pub fn dyn_op(&self) -> &dyn Op {
        &*self.obj
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> TraitOp<T> {
    pub fn try_from_op(op: Ptr<Operation>, ctx: &Context) -> Option<Self> {
        let op = op.dyn_op(ctx);
        if !op_impls::<T>(&*op) {
            None
        } else {
            Some(TraitOp {
                obj: op,
                _marker: PhantomData,
            })
        }
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> TryFrom<OpObj> for TraitOp<T> {
    type Error = pliron::result::Error;

    fn try_from(value: OpObj) -> Result<Self> {
        if !op_impls::<T>(&*value) {
            verify_err_noloc!("Op doesn't implement trait")
        } else {
            Ok(TraitOp {
                obj: value,
                _marker: PhantomData,
            })
        }
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> Deref for TraitOp<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        op_cast(self.obj.as_ref()).unwrap()
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> Eq for TraitOp<T> {}
impl<T: OpInterfaceMarker + 'static + ?Sized> PartialEq for TraitOp<T> {
    fn eq(&self, other: &Self) -> bool {
        self.obj == other.obj && self._marker == other._marker
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> Hash for TraitOp<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.obj.hash(state);
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> Copy for TraitOp<T> {}
impl<T: OpInterfaceMarker + 'static + ?Sized> Clone for TraitOp<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: OpInterfaceMarker + 'static + ?Sized> Debug for TraitOp<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let op = self.obj.get_operation();
        Debug::fmt(&op, f)
    }
}

pub(crate) fn operand_range_to_uses(
    ctx: &Context,
    op: Ptr<Operation>,
    range: Range<usize>,
) -> Vec<Use<Value>> {
    let op = op.deref(ctx);
    range.map(|idx| op.get_operand_as_use(idx)).collect()
}

#[macro_export]
macro_rules! small_map {
    {$($k: expr => $v: expr),* $(,)?} => {{
        let mut out = $crate::pliron::utils::table::SmallMap::new();
        $(out.insert($k, $v);)*
        out
    }};
}

#[macro_export]
macro_rules! small_set {
    {$($v: expr),* $(,)?} => {{
        let mut out = $crate::pliron::utils::table::SmallSet::new();
        $(out.insert($v);)*
        out
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AddressType,
        attributes::BoolAttr,
        dialect::branch::IfOp,
        scope::Scope,
        settings::{Dim3, ExecutionMode, KernelSettings},
    };
    use alloc::{boxed::Box, vec};
    use pliron::{
        basic_block::BasicBlock,
        builtin::op_interfaces::OneResultInterface,
        combine::stream::position::SourcePosition,
        location::{Located, Source},
    };

    /// A kernel scope without debug data, so that only the test gives locations.
    fn kernel() -> Scope {
        Scope::root(KernelSettings::new(
            Dim3::new_single(),
            ExecutionMode::Checked,
            AddressType::U32,
        ))
    }

    fn entry_block(scope: &Scope) -> Ptr<BasicBlock> {
        scope.state().entry_func.get_entry_block(scope.ctx())
    }

    /// Line `line` of the file `k.rs`.
    fn at(ctx: &mut Context, line: i32) -> Location {
        Location::SrcPos {
            src: Source::new_from_file(ctx, "k.rs"),
            pos: SourcePosition { line, column: 1 },
        }
    }

    /// A new constant at the end of `block`, at `loc`.
    fn constant(ctx: &mut Context, block: Ptr<BasicBlock>, loc: Location) -> Ptr<Operation> {
        let op = ConstantOp::new(ctx, Box::new(BoolAttr::new(true))).get_operation();
        op.deref_mut(ctx).set_loc(loc);
        op.insert_at_back(block, ctx);
        op
    }

    /// An op without a location gets the location of the op before it. The first op of a block
    /// gets the location of the op that holds the block. Without a location before it, an op
    /// keeps none.
    #[test]
    fn ops_inherit_the_location_before_them() {
        let scope = kernel();
        let ctx = scope.ctx_mut();
        let block = entry_block(&scope);
        let leading = block
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("the flag of the kernel");
        let (one, two, three) = (at(ctx, 1), at(ctx, 2), at(ctx, 3));

        let located = constant(ctx, block, one.clone());
        let after = constant(ctx, block, Location::Unknown);
        let cond = ConstantOp::new(ctx, Box::new(BoolAttr::new(true))).get_result(ctx);
        let if_op = IfOp::new(ctx, cond);
        if_op.get_operation().insert_at_back(block, ctx);
        let first_then = constant(ctx, if_op.then_block(ctx), Location::Unknown);
        let own = constant(ctx, if_op.then_block(ctx), two.clone());
        let after_own = constant(ctx, if_op.then_block(ctx), Location::Unknown);
        let first_else = constant(ctx, if_op.else_block(ctx), Location::Unknown);
        let last = constant(ctx, block, three.clone());

        let module = scope.state().module.get_operation();
        let mut analyses = AnalysisManager::default();
        let result = InheritLocationPass.run(module, ctx, &mut analyses).unwrap();
        assert!(matches!(result.ir_changed, IRStatus::Changed));

        assert!(leading.deref(ctx).loc().is_unknown());
        for (op, loc) in [
            (located, &one),
            (after, &one),
            (if_op.get_operation(), &one),
            (first_then, &one),
            (own, &two),
            (after_own, &two),
            (first_else, &one),
            (last, &three),
        ] {
            assert_eq!(op.deref(ctx).loc(), *loc);
        }

        let again = InheritLocationPass.run(module, ctx, &mut analyses).unwrap();
        assert!(matches!(again.ir_changed, IRStatus::Unchanged));
    }

    /// Replaces `target` with a new constant. It also inserts one more constant, a constant at
    /// its own location, and a constant that it erases again.
    struct SplitConstant {
        target: Ptr<Operation>,
        own: Location,
        inserted: Vec<Ptr<Operation>>,
    }

    /// A new constant at the insertion point of `rewriter`, without a location.
    fn insert(ctx: &mut Context, rewriter: &mut MatchRewriter) -> Ptr<Operation> {
        let op = ConstantOp::new(ctx, Box::new(BoolAttr::new(false))).get_operation();
        rewriter.insert_operation(ctx, op);
        op
    }

    impl MatchRewrite for SplitConstant {
        fn r#match(&mut self, _ctx: &Context, op: Ptr<Operation>) -> bool {
            op == self.target
        }

        fn rewrite(
            &mut self,
            ctx: &mut Context,
            rewriter: &mut MatchRewriter,
            op: Ptr<Operation>,
        ) -> Result<()> {
            let extra = insert(ctx, rewriter);
            let own = insert(ctx, rewriter);
            own.deref_mut(ctx).set_loc(self.own.clone());
            let erased = insert(ctx, rewriter);
            let replacement = insert(ctx, rewriter);
            let value = replacement.deref(ctx).get_result(0);
            rewriter.replace_operation_with_values(ctx, op, vec![value]);
            rewriter.erase_operation(ctx, erased);
            self.inserted = vec![extra, own, replacement];
            Ok(())
        }
    }

    /// Each op that a rewrite inserts without a location gets the location of the op it
    /// rewrites, not only the op that replaces it. An op with its own location keeps it, and an
    /// op that the rewrite erased is skipped.
    #[test]
    fn rewrites_keep_the_location_of_the_op() {
        let scope = kernel();
        let ctx = scope.ctx_mut();
        let block = entry_block(&scope);
        let (one, two) = (at(ctx, 1), at(ctx, 2));
        let target = constant(ctx, block, one.clone());

        let mut rewrite = SplitConstant {
            target,
            own: two.clone(),
            inserted: Vec::new(),
        };
        let module = scope.state().module.get_operation();
        apply_match_rewrite(
            ctx,
            &mut KeepLocation(&mut rewrite),
            RewriterOrder::default(),
            module,
        )
        .unwrap();

        let locations = rewrite
            .inserted
            .iter()
            .map(|op| op.deref(ctx).loc())
            .collect::<Vec<_>>();
        assert_eq!(locations, [one.clone(), two, one]);
    }
}
