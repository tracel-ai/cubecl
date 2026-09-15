use core::ops::{BitAndAssign, BitOrAssign};

use alloc::{boxed::Box, vec::Vec};
use cubecl_ir::interfaces::MemoryEffect;
use pliron::{context::Context, value::Value};

pub mod address_space;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum ModRefResult {
    NoModRef,
    Ref,
    Mod,
    ModRef,
}

impl ModRefResult {
    pub fn join(&self, other: ModRefResult) -> ModRefResult {
        match (*self, other) {
            (ModRefResult::NoModRef, other) | (other, ModRefResult::NoModRef) => other,
            (_, ModRefResult::ModRef) | (ModRefResult::ModRef, _) => ModRefResult::ModRef,
            (ModRefResult::Ref, ModRefResult::Ref) => ModRefResult::Ref,
            (ModRefResult::Mod, ModRefResult::Mod) => ModRefResult::Mod,
            (ModRefResult::Ref, ModRefResult::Mod) | (ModRefResult::Mod, ModRefResult::Ref) => {
                ModRefResult::ModRef
            }
        }
    }

    pub fn meet(&self, other: ModRefResult) -> ModRefResult {
        match (*self, other) {
            (ModRefResult::NoModRef, _) | (_, ModRefResult::NoModRef) => ModRefResult::NoModRef,
            (ModRefResult::ModRef, other) | (other, ModRefResult::ModRef) => other,
            (ModRefResult::Ref, ModRefResult::Ref) => ModRefResult::Ref,
            (ModRefResult::Mod, ModRefResult::Mod) => ModRefResult::Mod,
            (ModRefResult::Mod, ModRefResult::Ref) | (ModRefResult::Ref, ModRefResult::Mod) => {
                ModRefResult::NoModRef
            }
        }
    }

    pub fn contains_mod(&self) -> bool {
        match self {
            ModRefResult::NoModRef | ModRefResult::Ref => false,
            ModRefResult::Mod | ModRefResult::ModRef => true,
        }
    }
}

impl BitOrAssign for ModRefResult {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.join(rhs);
    }
}

impl BitAndAssign for ModRefResult {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.meet(rhs);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AliasResult {
    NoAlias,
    MayAlias,
    PartialAlias,
    MustAlias,
}

impl AliasResult {
    pub fn join(&self, other: AliasResult) -> AliasResult {
        if *self == other {
            return other;
        }
        match (self, other) {
            (AliasResult::PartialAlias, AliasResult::MustAlias)
            | (AliasResult::MustAlias, AliasResult::PartialAlias) => AliasResult::PartialAlias,
            _ => AliasResult::MayAlias,
        }
    }
}

pub trait AliasAnalysis {
    fn alias(&self, ctx: &Context, lhs: Value, rhs: Value) -> AliasResult;
    fn mod_ref(&self, ctx: &Context, location: &MemoryEffect, rhs: &MemoryEffect) -> ModRefResult;
}

#[derive(Default)]
pub struct AliasAnalysisStack {
    analyses: Vec<Box<dyn AliasAnalysis>>,
}

impl AliasAnalysisStack {
    pub fn add_analysis(&mut self, analysis: impl AliasAnalysis + 'static) {
        self.analyses.push(Box::new(analysis));
    }

    pub fn alias(&self, ctx: &Context, lhs: Value, rhs: Value) -> AliasResult {
        let mut result = AliasResult::MayAlias;
        for analysis in self.analyses.iter() {
            result = analysis.alias(ctx, lhs, rhs);
            if result != AliasResult::MayAlias {
                return result;
            }
        }
        result
    }

    pub fn mod_ref<'a>(
        &self,
        ctx: &Context,
        location: impl IntoIterator<Item = &'a MemoryEffect> + Clone,
        rhs: impl IntoIterator<Item = &'a MemoryEffect> + Clone,
    ) -> ModRefResult {
        // Within an effect pair: meet AA results, since each analysis refines the
        // conservative answer.
        //
        // Across effect pairs: join results, since either pair may account for the
        // interaction between the two effect sets.
        let mut result = ModRefResult::NoModRef;
        for lhs_effect in location.clone() {
            for rhs_effect in rhs.clone() {
                let mut effect_res = ModRefResult::ModRef;
                for analysis in self.analyses.iter() {
                    effect_res &= analysis.mod_ref(ctx, lhs_effect, rhs_effect);
                }
                result |= effect_res;
            }
        }

        result
    }
}

pub(crate) fn effect_mod_ref(effect: &MemoryEffect) -> ModRefResult {
    match effect {
        MemoryEffect::Read(_) | MemoryEffect::ReadAllInSpace(_) | MemoryEffect::ReadAll => {
            ModRefResult::Ref
        }
        MemoryEffect::Write(_) | MemoryEffect::WriteAllInSpace(_) | MemoryEffect::WriteAll => {
            ModRefResult::Mod
        }
        MemoryEffect::Opaque => ModRefResult::ModRef,
    }
}
