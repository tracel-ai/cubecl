use crate::{
    analyses::alias_analysis::{AliasAnalysis, AliasResult, ModRefResult, effect_mod_ref},
    passes::alloc_shared_memory::SliceSharedOp,
};
use cubecl_ir::{
    interfaces::{MemoryEffect, aliasing::PointerExt},
    prelude::*,
};
use pliron::opts::mem2reg::PromotableAllocationInterface;

/// Look at the root allocation to determine if two pointers are non-overlapping.
/// Explicitly does not handle kernel params because those are handled by address spaces (each
/// global space is unique). Delegates to the existing root pointer machinery so anything more
/// refined like interprocedural tracking can be done there.
///
/// Assumes shared slices are non-aliasing since that's part of the contract (no two overlapping
/// slices may be live at the same time).
pub struct RootAllocAA;

impl AliasAnalysis for RootAllocAA {
    fn alias(&self, ctx: &Context, lhs: Value, rhs: Value) -> AliasResult {
        if lhs == rhs {
            return AliasResult::MustAlias;
        }

        let Some(lhs_root) = lhs.get_root_defining_op(ctx) else {
            return AliasResult::MayAlias;
        };
        let Some(rhs_root) = rhs.get_root_defining_op(ctx) else {
            return AliasResult::MayAlias;
        };
        let lhs_is_alloc = lhs_root.impls::<dyn PromotableAllocationInterface>(ctx)
            || lhs_root.is_op::<SliceSharedOp>(ctx);
        let rhs_is_alloc = rhs_root.impls::<dyn PromotableAllocationInterface>(ctx)
            || rhs_root.is_op::<SliceSharedOp>(ctx);

        if lhs_is_alloc && rhs_is_alloc {
            let root_value = lhs.get_root_value(ctx);
            let root_type = root_value.get_type(ctx);

            if lhs_root == rhs_root && root_value == rhs.get_root_value(ctx) {
                // If the full type is accessed, the pointers must alias
                if root_type == lhs.get_type(ctx) && root_type == rhs.get_type(ctx) {
                    AliasResult::MustAlias
                } else {
                    AliasResult::MayAlias
                }
            } else {
                AliasResult::NoAlias
            }
        } else {
            AliasResult::MayAlias
        }
    }

    fn mod_ref(&self, ctx: &Context, location: &MemoryEffect, rhs: &MemoryEffect) -> ModRefResult {
        let Some((lhs_val, rhs_val)) = location.value().zip(rhs.value()) else {
            return ModRefResult::ModRef;
        };
        match self.alias(ctx, lhs_val, rhs_val) {
            AliasResult::NoAlias => ModRefResult::NoModRef,
            AliasResult::MayAlias | AliasResult::PartialAlias | AliasResult::MustAlias => {
                effect_mod_ref(rhs)
            }
        }
    }
}
