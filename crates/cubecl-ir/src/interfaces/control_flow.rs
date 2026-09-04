use crate::{dialect::RegionPtrExt, prelude::*};
use derive_more::From;
use pliron::{
    attribute::AttrObj, builtin::ops::FuncOp, linked_list::ContainsLinkedList, region::Region,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionPredecessor {
    Parent,
    Terminator(TraitOpPtr<dyn RegionBranchTerminatorOpInterface>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, From)]
pub enum RegionSuccessor {
    Region(Ptr<Region>),
    AfterOp,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InvocationBounds {
    lower_bound: usize,
    upper_bound: Option<usize>,
}

impl InvocationBounds {
    pub fn unknown() -> Self {
        Self {
            lower_bound: 0,
            upper_bound: None,
        }
    }

    pub fn never() -> Self {
        Self {
            lower_bound: 0,
            upper_bound: Some(0),
        }
    }

    pub fn once() -> Self {
        Self {
            lower_bound: 1,
            upper_bound: Some(1),
        }
    }

    pub fn zero_or_one() -> Self {
        Self {
            lower_bound: 0,
            upper_bound: Some(1),
        }
    }
}

#[op_interface]
pub trait CallableOpInterface {
    verify_op_succ!();
    fn callable_region(&self, ctx: &Context) -> Option<Ptr<Region>>;
    fn argument_types(&self, ctx: &Context) -> Vec<TypeHandle>;
    fn result_types(&self, ctx: &Context) -> Vec<TypeHandle>;
}

#[op_interface_impl]
impl CallableOpInterface for FuncOp {
    fn callable_region(&self, ctx: &Context) -> Option<Ptr<Region>> {
        Some(self.get_region(ctx))
    }

    fn argument_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        let ty = self.get_attr_func_type(ctx).unwrap().get_type(ctx);
        type_cast::<dyn FunctionTypeInterface>(&*ty.deref(ctx))
            .unwrap()
            .arg_types()
    }

    fn result_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        let ty = self.get_attr_func_type(ctx).unwrap().get_type(ctx);
        type_cast::<dyn FunctionTypeInterface>(&*ty.deref(ctx))
            .unwrap()
            .res_types()
    }
}

#[format]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SymbolVisiblity {
    Public,
    Private,
    Nested,
}

#[op_interface]
pub trait SymbolOpInterface: pliron::builtin::op_interfaces::SymbolOpInterface {
    verify_op_succ!();

    fn get_visibility(&self, ctx: &Context) -> SymbolVisiblity;
    fn set_visibility(&self, ctx: &mut Context, visibility: SymbolVisiblity);

    fn is_public(&self, ctx: &Context) -> bool {
        matches!(self.get_visibility(ctx), SymbolVisiblity::Public)
    }

    fn is_private(&self, ctx: &Context) -> bool {
        matches!(self.get_visibility(ctx), SymbolVisiblity::Private)
    }

    fn is_nested(&self, ctx: &Context) -> bool {
        matches!(self.get_visibility(ctx), SymbolVisiblity::Nested)
    }
}

/// This interface provides information for region-holding operations that
/// exhibit branching behavior between held regions. It models the control flow
/// edges between regions (and between the op and its regions), as well as the
/// data flow (value propagation) that occurs along those control flow edges.
///
/// This interface is meant to model well-defined cases of control-flow and
/// value propagation, where what occurs along control-flow edges is assumed to
/// be side-effect free.
///
/// A "region branch point" indicates the point from which a branch (edge)
/// originates. It can indicate:
/// 1. A `RegionBranchTerminatorOpInterface` terminator in any of the
///    immediately nested regions of this op.
/// 2. `RegionBranchPoint::parent()`: the branch originates from outside of the
///    op, i.e., when first executing this op.
///
/// When branching from a region branch point to a region successor, the
/// "successor operands" to be forwarded from the region branch point can be
/// specified with `getEntrySuccessorOperands` /
/// `RegionBranchTerminatorOpInterface::getSuccessorOperands`.
///
/// A "region successor" indicates the target of a branch. It can indicate:
/// 1. A region of this op.
/// 2. A parent operation, i.e., the control flow leaves/resumes after that op.
///
/// The SSA values to which successor operands are forwarded are called
/// "successor inputs".
///
/// By default, successor operands and successor block arguments/successor
/// results must have the same type. `areTypesCompatible` can be implemented to
/// allow non-equal types.
///
///
/// Note: This interface works in conjunction with
/// `RegionBranchTerminatorOpInterface`. All immediately nested block
/// terminators that model branching between regions must implement the
/// `RegionBranchTerminatorOpInterface`. Otherwise, analyses/transformations
/// may miss control flow edges and produce incorrect results. Not every block
/// terminator is necessarily a region branch terminator: e.g., in the presence
/// of unstructured control flow, a block terminator could indicate a branch to
/// a different block within the same region.
///
/// Example:
///
/// ```ignore
/// %r = scf.for %iv = %lb to %ub step %step iter_args(%a = %b)
///     -> tensor<5xf32> {
///   ...
///   scf.yield %c : tensor<5xf32>
/// }
/// ```
///
/// `scf.for` has one region. There are two region branch points with two
/// identical region successors:
/// * parent => op(%r), region0(%a)
/// * `scf.yield` => op(%r), region0(%a)
///
/// `%a` and %r are successor inputs. `%b` is an entry successor operand. `%c`
/// is a successor operand.
#[op_interface]
pub trait RegionBranchOpInterface {
    verify_op_succ!();

    /// Returns the operands of this operation that are forwarded to the
    /// successor inputs when branching to `successor`. `successor` is
    /// guaranteed to be among the successors that are returned by
    /// `entry_successor_regions`/`successor_regions(parent())`.
    ///
    /// Example: In the above example, this method returns the operand %b of the
    /// `scf.for` op, regardless of the value of `successor`. I.e., this op always
    /// forwards the same operands, regardless of whether the loop has 0 or more
    /// iterations.
    fn entry_successor_operands(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value> {
        let _ = (ctx, successor);
        vec![]
    }

    /// Returns all potential region successors when first executing the op.
    ///
    /// Unlike `successor_regions`, this method also receives the constant
    /// operands of this op (one entry per operand, `None` if the operand has
    /// no/unknown constant value). The implementation may use this information
    /// to filter out successors. By default, it simply dispatches to
    /// `successor_regions`.
    ///
    /// Note: The control flow does not necessarily have to enter any region of
    /// this op.
    ///
    /// Example: In the above example, this method may return two region
    /// region successors: the single region of the `scf.for` op and the
    /// `scf.for` operation (that implements this interface). If %lb, %ub, %step
    /// are constants and it can be determined the loop does not have any
    /// iterations, this method may choose to return only this operation.
    /// Similarly, if it can be determined that the loop has at least one
    /// iteration, this method may choose to return only the region of the loop.
    fn entry_successor_regions(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let _ = (ctx, operands);
        self.successor_regions(ctx, RegionPredecessor::Parent)
    }

    /// Returns all potential region successors when branching from `predecessor`.
    /// These are the regions that may be selected during the flow of control.
    ///
    /// When `pred = RegionPredecessor::Parent`, this method returns the
    /// region successors when entering the operation. Otherwise, this method
    /// returns the successor regions when branching from the region indicated
    /// by `pred`.
    ///
    /// Example: In the above example, this method returns the region of the
    /// `scf.for` and `parent` for either region branch point. An implementation
    /// may choose to filter out region successors when it is statically known
    /// (e.g., by examining the operands of this op) that those successors are
    /// not branched to.
    fn successor_regions(&self, ctx: &Context, pred: RegionPredecessor) -> Vec<RegionSuccessor>;

    /// Returns all potential region successors when branching from any
    /// terminator in `region`.
    fn all_successor_regions_of_region(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
    ) -> Vec<RegionSuccessor> {
        let mut res = vec![];
        for block in region.deref(ctx).iter(ctx) {
            let Some(term) = block.deref(ctx).get_terminator(ctx) else {
                continue;
            };
            if let Some(terminator) = TraitOpPtr::try_from_op(term, ctx) {
                res.extend(self.successor_regions(ctx, RegionPredecessor::Terminator(terminator)));
            }
        }
        res
    }

    /// Return all successor inputs for the given region successor. If the
    /// given region successor is a region, then the returned values are block
    /// arguments. Otherwise, if the given region successor is an operation,
    /// the returned values are op results.
    fn successor_inputs(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value>;

    /// Returns all potential branching points (predecessors) for a given
    /// region successor.
    fn predecessors(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<RegionPredecessor> {
        let mut predecessors = vec![];
        for pred in all_region_predecessors(self.get_operation(), ctx) {
            let successors = self.successor_regions(ctx, pred);
            let is_pred = successors.iter().any(|succ| succ == &successor);
            if is_pred {
                predecessors.push(pred);
            }
        }
        predecessors
    }

    /// Returns all potential values across all (predecessors) for a given successor
    /// input.
    fn predecessor_values(
        &self,
        ctx: &Context,
        successor: RegionSuccessor,
        index: usize,
    ) -> Vec<Value> {
        let mut predecessor_values = vec![];
        let predecessors = self.predecessors(ctx, successor);
        for predecessor in predecessors {
            match predecessor {
                RegionPredecessor::Parent => {
                    predecessor_values.push(self.entry_successor_operands(ctx, successor)[index]);
                }
                RegionPredecessor::Terminator(term) => {
                    predecessor_values
                        .push(term.deref(ctx).successor_operands(ctx, successor)[index]);
                }
            }
        }
        predecessor_values
    }

    /// Populates `invocationBounds` with the minimum and maximum number of
    /// times this operation will invoke the attached regions (assuming the
    /// regions yield normally, i.e. do not abort or invoke an infinite loop).
    /// The minimum number of invocations is at least 0. If the maximum number
    /// of invocations cannot be statically determined, then it will be set to
    /// `InvocationBounds::getUnknown()`.
    ///
    /// This method also passes along the constant operands of this op.
    /// `operands` contains an entry for every operand of this op, with a null
    /// attribute if the operand has no constant value.
    ///
    /// This method may be called speculatively on operations where the provided
    /// operands are not necessarily the same as the operation's current
    /// operands. This may occur in analyses that wish to determine "what would
    /// be the region invocations if these were the operands?"
    fn region_invocation_bounds(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<InvocationBounds> {
        let _ = (ctx, operands);
        vec![InvocationBounds::unknown()]
    }

    /// This method is called to compare types along control-flow edges. By
    /// default, the types are checked as equal.
    fn are_types_compatible(&self, ctx: &Context, lhs: TypeHandle, rhs: TypeHandle) -> bool {
        let _ = ctx;
        lhs == rhs
    }
}

impl dyn RegionBranchOpInterface {
    /// Return the successor operands from the source branch point to the
    /// destination region successor.
    ///
    /// If the branch point is the parent op, this function returns entry
    /// successor operands of this op. Otherwise, it returns successor operands
    /// of the respective terminator.
    pub fn successor_operands(
        &self,
        ctx: &Context,
        src: RegionPredecessor,
        dest: RegionSuccessor,
    ) -> Vec<Value> {
        match src {
            RegionPredecessor::Parent => self.entry_successor_operands(ctx, dest),
            RegionPredecessor::Terminator(term) => term.deref(ctx).successor_operands(ctx, dest),
        }
    }

    /// Return all successor inputs for the given region successor.
    ///
    /// If the "successor" is a region, it will return non-forwarded arguments,
    /// if it is a "parent", it will return non-forwarded results.
    pub fn non_successor_inputs(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value> {
        let results = match successor {
            RegionSuccessor::Region(region) => region.arguments(ctx),
            RegionSuccessor::AfterOp => {
                let parent = self.get_operation().deref(ctx).get_parent_op(ctx).unwrap();
                parent.results(ctx)
            }
        };
        let successor_inputs = self.successor_inputs(ctx, successor);
        results
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !successor_inputs.iter().any(|it| it.find_index(ctx) == *i))
            .map(|it| it.1)
            .collect()
    }
    /// Return all possible region branch points: the region branch op itself
    /// and all region branch terminators.
    pub fn all_region_predecessors(&self, ctx: &Context) -> Vec<RegionPredecessor> {
        all_region_predecessors(self.get_operation(), ctx)
    }
}

pub fn all_region_predecessors(op: Ptr<Operation>, ctx: &Context) -> Vec<RegionPredecessor> {
    let mut predecessors = vec![RegionPredecessor::Parent];
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            if let Some(term) = block.deref(ctx).get_terminator(ctx)
                && let Some(term) =
                    TraitOpPtr::<dyn RegionBranchTerminatorOpInterface>::try_from_op(term, ctx)
            {
                predecessors.push(RegionPredecessor::Terminator(term));
            }
        }
    }
    predecessors
}

/// This interface provides information for branching terminator operations
/// in the presence of a parent `RegionBranchOpInterface` implementation. It
/// acts as a marker for valid region branch points and specifies which
/// operands are passed to which region successor.
///
/// Note: If an operation does not implement the
/// `RegionBranchTerminatorOpInterface`, then that op has no region successors.
/// (However, there may be other block terminators in the same region that
/// implement the `RegionBranchTerminatorOpInterface`, so the enclosing region
/// may have region successors.)
#[op_interface]
pub trait RegionBranchTerminatorOpInterface {
    verify_op_succ!();

    /// Returns a range of operands that are semantically "returned" by passing
    /// them to the region successor.
    fn successor_operands(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value>;

    /// Returns all potential region successors that are branched to after this
    /// terminator based on the given constant operands.
    ///
    /// This method also receives the constant operands of this op (one entry
    /// per operand, `None` if the operand has no/unknown constant value). The
    /// implementation may use this information to filter out successors.
    /// By default, it simply dispatches to the parent
    /// `RegionBranchOpInterface`'s `successor_regions` implementation.
    fn successor_regions(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let _ = operands;
        let op = TraitOpPtr::try_from_op(self.get_operation(), ctx).unwrap();
        let parent = self.get_operation().deref(ctx).get_parent_op(ctx).unwrap();
        let parent = parent.dyn_op(ctx);
        op_cast::<dyn RegionBranchOpInterface>(&*parent)
            .unwrap()
            .successor_regions(ctx, RegionPredecessor::Terminator(op))
    }
}
