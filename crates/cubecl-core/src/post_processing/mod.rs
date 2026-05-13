pub mod analysis_helper;
pub mod checked_io;
pub mod dead_code;
pub mod disaggregate;
pub mod expression_merge;
pub mod predicate;
pub mod saturating;
pub mod unroll;
pub mod util;

// Shared passes
pub mod constant_prop;

pub mod visitor;

use cubecl_ir::Scope;

use crate::post_processing::{
    analysis_helper::{BufferVisibility, GlobalAnalyses},
    constant_prop::{ConstEval, ConstOperandSimplify},
    dead_code::EliminateUnusedExpressions,
    expression_merge::InlineAssignments,
    util::AtomicCounter,
    visitor::InstructionVisitor,
};

pub fn optimize_scope(scope: &Scope) -> BufferVisibility {
    let analyses = GlobalAnalyses::default();
    analyses.recalculate_pointer_source(scope);

    let changes = AtomicCounter::new(1);
    while changes.get_and_reset() != 0 {
        ConstOperandSimplify.visit_scope(scope, &analyses, &changes);
        ConstEval.visit_scope(scope, &analyses, &changes);
        ConstEval.visit_scope(scope, &analyses, &changes);
        InlineAssignments::default().visit_scope(scope, &analyses, &changes);

        analyses.recalculate_used_values(scope);
        EliminateUnusedExpressions.visit_scope(scope, &analyses, &changes);
    }

    BufferVisibility::new(scope, &analyses)
}
