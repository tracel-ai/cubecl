use alloc::boxed::Box;
use cubecl_ir::{
    AddressSpace, NamedRewrite,
    attributes::IndexAttr,
    dialect::{
        general::ReinterpretCastOp,
        math::{IAddOp, IMulOp},
        memory::{IndexOp, LoadOp},
        vector::VectorExtractDynamicOp,
    },
    interfaces::TypedExt as _,
    prelude::{Inserter as _, Rewriter as _, *},
    rewrite::MatchRewritePass,
    types::{ArrayType, PointerType, RuntimeArrayType},
};
use pliron::{builtin::ops::ConstantOp, linked_list::LinkedList as _, opts::dce::SideEffects};

/// Read a vector's lane at a run-time index straight out of the memory the vector was loaded
/// from, rather than through the whole vector.
///
/// A target whose vectors are structs (C++, MSL) picks a lane known only at run time with a chain
/// of selects, one a lane, where the memory holding the vector serves that lane alone with one
/// scalar load. Targets that cannot view a buffer at another width (WGSL) leave this pass out.
pub type ScalarExtractPass = MatchRewritePass<ScalarExtract>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct ScalarExtract;

/// The element pointer and the load of a vector `op` extracts a lane of at a run-time index, when
/// the lane can be read alone in its place.
///
/// The vector must be loaded from an element of a list in global or shared memory, a local one
/// staying in registers, and serve only such extracts, so no load of the whole vector remains.
/// The lane's load lands where `op` is, so the ops between the vector's load and `op` must not
/// write anything.
fn lane_source(ctx: &Context, op: Ptr<Operation>) -> Option<(IndexOp, LoadOp)> {
    let extract = op.as_op::<VectorExtractDynamicOp>(ctx)?;
    let load = extract.vector(ctx).defining_op()?.as_op::<LoadOp>(ctx)?;
    let element = load.ptr(ctx).defining_op()?.as_op::<IndexOp>(ctx)?;
    let list = element.base(ctx).get_type(ctx).deref(ctx);
    let PointerType {
        inner,
        address_space,
    } = list.downcast_ref::<PointerType>()?;
    if *address_space == AddressSpace::Local {
        return None;
    }
    let list = inner.deref(ctx);
    if !list.is::<ArrayType>() && !list.is::<RuntimeArrayType>() {
        return None;
    }

    let vector = load.get_result(ctx);
    let only_extracts = vector.uses(ctx).iter().all(|r#use| {
        r#use
            .user_op()
            .as_op::<VectorExtractDynamicOp>(ctx)
            .is_some()
    });
    if !only_extracts {
        return None;
    }

    let block = load.get_operation().deref(ctx).get_parent_block()?;
    if op.deref(ctx).get_parent_block()? != block {
        return None;
    }
    let mut between = load.get_operation().deref(ctx).get_next();
    while let Some(next) = between {
        if next == op {
            return Some((element, load));
        }
        let writes = next.deref(ctx).num_regions() > 0
            || op_cast::<dyn SideEffects>(&*next.dyn_op(ctx))
                .is_none_or(|effects| effects.has_side_effects(ctx));
        if writes {
            return None;
        }
        between = next.deref(ctx).get_next();
    }
    None
}

/// `ptr`, a pointer to a list of vectors, as a pointer to the same memory as a list of their
/// scalars.
fn scalar_list(ctx: &mut Context, ptr: Value) -> TypeHandle {
    let vector_size = ptr.vector_size(ctx);
    let PointerType {
        inner,
        address_space,
    } = *ptr.get_type(ctx).deref(ctx).downcast_ref().unwrap();
    let scalar = inner.scalar_ty(ctx);
    let list = inner.deref(ctx);
    let scalars = if let Some(ArrayType { length, .. }) = list.downcast_ref::<ArrayType>().copied()
    {
        ArrayType::get(ctx, scalar, length * vector_size).into()
    } else {
        RuntimeArrayType::get(ctx, scalar).into()
    };
    PointerType::get(ctx, scalars, address_space).into()
}

impl MatchRewrite for ScalarExtract {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        lane_source(ctx, op).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some((element, load)) = lane_source(ctx, op) else {
            return Ok(());
        };
        let extract = op.as_op::<VectorExtractDynamicOp>(ctx).unwrap();
        let list = element.base(ctx);
        let vector_size = load.get_result(ctx).vector_size(ctx);

        let scalars_ty = scalar_list(ctx, list);
        let scalars = ReinterpretCastOp::new(ctx, scalars_ty, list);
        rewriter.insert_op(ctx, &scalars);
        let width = ConstantOp::new(ctx, Box::new(IndexAttr::new(vector_size)));
        rewriter.insert_op(ctx, &width);
        let first = IMulOp::new(ctx, element.index(ctx), width.get_result(ctx));
        rewriter.insert_op(ctx, &first);
        let at = IAddOp::new(ctx, first.get_result(ctx), extract.index(ctx));
        rewriter.insert_op(ctx, &at);
        let checked = element.checked(ctx);
        let lane =
            IndexOp::maybe_checked(ctx, scalars.get_result(ctx), at.get_result(ctx), checked);
        rewriter.insert_op(ctx, &lane);
        let value = LoadOp::new(ctx, lane.get_result(ctx));
        rewriter.insert_op(ctx, &value);
        rewriter.replace_operation(ctx, op, value.get_operation());
        Ok(())
    }
}
