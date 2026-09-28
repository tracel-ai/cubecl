use cubecl_ir::{AddressSpace, interfaces::MemoryEffect, prelude::Value, types::PointerType};
use pliron::{context::Context, r#type::Typed};

use crate::analyses::alias_analysis::{AliasAnalysis, effect_mod_ref};

use super::{AliasResult, ModRefResult};

pub struct AddressSpaceAA;

impl AliasAnalysis for AddressSpaceAA {
    fn alias(&self, ctx: &Context, lhs: Value, rhs: Value) -> AliasResult {
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
        let Some(location_space) = effect_address_space(ctx, location) else {
            return ModRefResult::ModRef;
        };
        let Some(rhs_space) = effect_address_space(ctx, rhs) else {
            return ModRefResult::ModRef;
        };

        if location_space != rhs_space {
            ModRefResult::NoModRef
        } else {
            effect_mod_ref(rhs)
        }
    }
}

fn effect_address_space(ctx: &Context, effect: &MemoryEffect) -> Option<AddressSpace> {
    match effect {
        MemoryEffect::Read(value) if let Some(space) = ptr_address_space(ctx, *value) => {
            Some(space)
        }
        MemoryEffect::Write(value) if let Some(space) = ptr_address_space(ctx, *value) => {
            Some(space)
        }
        MemoryEffect::ReadAllInSpace(space) | MemoryEffect::WriteAllInSpace(space) => Some(*space),
        MemoryEffect::Read(_) | MemoryEffect::Write(_) => None,
        MemoryEffect::ReadAll | MemoryEffect::WriteAll | MemoryEffect::Opaque => None,
    }
}

fn ptr_address_space(ctx: &Context, value: Value) -> Option<AddressSpace> {
    let ty = value.get_type(ctx).deref(ctx);
    Some(ty.downcast_ref::<PointerType>()?.address_space)
}
