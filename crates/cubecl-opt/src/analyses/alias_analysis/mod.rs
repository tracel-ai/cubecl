use core::cell::RefMut;

use alloc::{boxed::Box, vec::Vec};
use cubecl_ir::{
    interfaces::side_effects::{MemoryEffect, ModRefResult},
    prelude::*,
};
use pliron::{context::Context, pass::AnalysisManager, value::Value};

use crate::analyses::{
    alias_analysis::{address_space::AddressSpaceAA, root_alloc::RootAllocAA},
    memory_ssa::MemorySSA,
};

pub mod address_space;
pub mod root_alloc;

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

pub fn default_stack() -> AliasAnalysisStack {
    let mut stack = AliasAnalysisStack::default();
    stack.add_analysis(AddressSpaceAA);
    stack.add_analysis(RootAllocAA);
    stack
}

pub fn default_memory_ssa<'a>(
    ctx: &Context,
    op: Ptr<Operation>,
    analyses: &'a mut AnalysisManager,
) -> Result<RefMut<'a, MemorySSA>> {
    let mut memory_ssa = analyses.get_analysis_mut::<MemorySSA>(op, ctx)?;
    memory_ssa.set_alias_analysis_stack(default_stack());
    Ok(memory_ssa)
}
