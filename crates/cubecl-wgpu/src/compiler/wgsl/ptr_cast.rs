//! Loads and stores through a pointer cast, which WGSL can't express: `*cast(p)` becomes
//! `bitcast(*p)` and `*cast(p) = v` becomes `*p = bitcast(v)`.
//!
//! Buffer record fields are such pointers: a word of the record, cast to the field's type (see
//! `cubecl_core::frontend::RecordField`). A WGSL record field must be 4 bytes, the size of the
//! word.

use cubecl_ir::{
    NamedRewrite, Scope,
    dialect::{
        general::ReinterpretCastOp,
        memory::{LoadOp, StoreOp},
    },
    interfaces::TypedExt,
    prelude::*,
};
use pliron::{input_err_noloc, result::Result};
use thiserror::Error;

pub type LowerPtrCastPass = MatchRewritePass<LowerPtrCast>;

#[derive(Default, NamedRewrite)]
pub struct LowerPtrCast;

#[derive(Debug, Error)]
#[error(
    "WGSL has no pointer casts: a {0} byte buffer record field can't be read from 4 byte words, \
     WGSL record fields are f32, u32 or i32"
)]
pub struct RecordFieldUnsupported(usize);

/// The pointer `ptr` was cast from, when `ptr` is a pointer cast.
fn cast_source(ctx: &Context, ptr: Value) -> Option<Value> {
    let cast = ptr.defining_op()?.as_op::<ReinterpretCastOp>(ctx)?;
    let source = cast.input(ctx);
    source.is_ptr(ctx).then_some(source)
}

/// The pointer a load or store accesses.
fn accessed_ptr(ctx: &Context, op: Ptr<Operation>) -> Option<Value> {
    if let Some(load) = op.as_op::<LoadOp>(ctx) {
        Some(load.ptr(ctx))
    } else {
        op.as_op::<StoreOp>(ctx).map(|store| store.ptr(ctx))
    }
}

impl MatchRewrite for LowerPtrCast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        accessed_ptr(ctx, op).is_some_and(|ptr| cast_source(ctx, ptr).is_some())
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let ptr = accessed_ptr(ctx, op).expect("matched a load or store");
        let source = cast_source(ctx, ptr).expect("matched a pointer cast");
        let source_ty = source.unwrap_ptr(ctx);
        let cast_ty = ptr.unwrap_ptr(ctx);
        if source_ty.size(ctx) != cast_ty.size(ctx) {
            return input_err_noloc!(RecordFieldUnsupported(cast_ty.size(ctx)));
        }

        let store_value = op.as_op::<StoreOp>(ctx).map(|store| store.value(ctx));
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let new_values = match store_value {
            Some(value) => {
                let value = scope.register_with_result(&ReinterpretCastOp::new(
                    scope.ctx_mut(),
                    source_ty,
                    value,
                ));
                scope.register(&StoreOp::new(scope.ctx_mut(), source, value));
                vec![]
            }
            None => {
                let raw = scope.register_with_result(&LoadOp::new(scope.ctx_mut(), source));
                let value = scope.register_with_result(&ReinterpretCastOp::new(
                    scope.ctx_mut(),
                    cast_ty,
                    raw,
                ));
                vec![value]
            }
        };
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}
