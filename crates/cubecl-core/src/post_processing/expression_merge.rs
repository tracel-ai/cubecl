use alloc::{vec, vec::Vec};
use cubecl_ir::{CoopMma, GlobalState, Instruction, Operation, Operator, UnaryOperands, Variable};
use hashbrown::HashMap;

use crate::post_processing::{
    analysis_helper::GlobalAnalyses,
    util::AtomicCounter,
    visitor::{InstructionVisitor, Visitor},
};

/// Inline constants or simple reassignments that don't change the type. This simplifies the code
/// and makes it easier to find optimizable expressions.
#[derive(Default, Debug)]
pub struct InlineAssignments {
    substitutions: HashMap<Variable, Variable>,
}

impl InstructionVisitor for InlineAssignments {
    fn visit_instruction(
        &mut self,
        mut inst: Instruction,
        _global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(());
        visitor.visit_operation(&mut inst.operation, analyses, |_, var| {
            if let Some(substitution) = self.substitutions.get(var) {
                *var = *substitution;
                changes.inc();
            }
        });

        match &mut inst.operation {
            Operation::Copy(input)
            | Operation::Operator(Operator::Cast(UnaryOperands { input }))
            | Operation::Operator(Operator::Reinterpret(UnaryOperands { input }))
            | Operation::CoopMma(CoopMma::Cast { input })
                if (input.is_immutable() || input.is_array() || input.ty.is_ptr())
                    && (inst.out.unwrap().is_immutable()
                        || inst.out.unwrap().is_array()
                        || inst.out.unwrap().ty.is_ptr())
                    && input.ty == inst.out.unwrap().ty =>
            {
                self.substitutions.insert(inst.out.unwrap(), *input);
                changes.inc();
                vec![]
            }
            _ => vec![inst],
        }
    }
}
