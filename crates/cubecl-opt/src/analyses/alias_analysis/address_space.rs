use cubecl_ir::{
    AddressSpace, EnumSet,
    interfaces::side_effects::{Extent, MemoryEffect, MemoryLocation, MemoryScope, ModRefResult},
    prelude::Value,
    types::PointerType,
};
use pliron::{context::Context, r#type::Typed};

use crate::analyses::alias_analysis::AliasAnalysis;

use super::AliasResult;

pub struct AddressSpaceAA;

impl AliasAnalysis for AddressSpaceAA {
    fn alias(
        &self,
        ctx: &Context,
        (lhs, _): (Value, Extent),
        (rhs, _): (Value, Extent),
    ) -> AliasResult {
        let Some(lhs_space) = ptr_address_space(ctx, lhs) else {
            return AliasResult::MayAlias;
        };
        let Some(rhs_space) = ptr_address_space(ctx, rhs) else {
            return AliasResult::MayAlias;
        };
        if lhs_space != rhs_space {
            AliasResult::NoAlias
        } else {
            AliasResult::MayAlias
        }
    }

    fn mod_ref(&self, ctx: &Context, location: &MemoryEffect, rhs: &MemoryEffect) -> ModRefResult {
        if any_effect_scoped(&[location, rhs]) {
            let Some(location_scope) = effect_memory_scope(ctx, location) else {
                return ModRefResult::ModRef;
            };
            let Some(rhs_scope) = effect_memory_scope(ctx, rhs) else {
                return ModRefResult::ModRef;
            };

            if location_scope.is_disjoint(rhs_scope) {
                ModRefResult::NoModRef
            } else {
                rhs.mod_ref_mask()
            }
        } else {
            let Some(location_space) = effect_address_space(ctx, location) else {
                return ModRefResult::ModRef;
            };
            let Some(rhs_space) = effect_address_space(ctx, rhs) else {
                return ModRefResult::ModRef;
            };

            if location_space != rhs_space {
                ModRefResult::NoModRef
            } else {
                rhs.mod_ref_mask()
            }
        }
    }
}

fn any_effect_scoped(effect: &[&MemoryEffect]) -> bool {
    effect
        .iter()
        .any(|it| matches!(it.location(), MemoryLocation::Scope(_)))
}

fn effect_address_space(ctx: &Context, effect: &MemoryEffect) -> Option<AddressSpace> {
    match effect.location() {
        MemoryLocation::Pointer(value, _) if let Some(space) = ptr_address_space(ctx, value) => {
            Some(space)
        }
        MemoryLocation::Pointer(..) | MemoryLocation::Scope(_) | MemoryLocation::All => None,
    }
}

fn effect_memory_scope(ctx: &Context, effect: &MemoryEffect) -> Option<EnumSet<MemoryScope>> {
    match effect.location() {
        MemoryLocation::Pointer(value, _) if let Some(space) = ptr_address_space(ctx, value) => {
            Some(space.memory_scope().into())
        }
        MemoryLocation::Pointer(..) => None,
        MemoryLocation::Scope(scope) => Some(scope),
        MemoryLocation::All => None,
    }
}

fn ptr_address_space(ctx: &Context, value: Value) -> Option<AddressSpace> {
    let ty = value.get_type(ctx).deref(ctx);
    Some(ty.downcast_ref::<PointerType>()?.address_space)
}
