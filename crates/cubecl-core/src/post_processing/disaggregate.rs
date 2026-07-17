use alloc::{format, vec::Vec};

use cubecl_environment::collections::HashMap;
use cubecl_ir::{GlobalState, Instruction, NonSemantic, Operation, Scope, Value, ValueKind};

use crate::post_processing::{
    analysis_helper::GlobalAnalyses,
    util::AtomicCounter,
    visitor::{InstructionVisitor, Visitor},
};

type Substitutes = HashMap<ValueKind, Vec<Value>>;
type Extracted = HashMap<ValueKind, Value>;

/// Disaggregates compiler-internal aggregates like bounds checked pointers and slice pointers into
/// individual variables.
#[derive(Debug, Default)]
pub struct DisaggregateVisitor {
    substitutes: Substitutes,
    extracted: Extracted,
}

impl DisaggregateVisitor {
    pub fn apply(scope: &Scope) {
        let mut this = Self::default();
        let changes = AtomicCounter::new(0);
        // We don't care about pointer sources or used variables at this point
        let analyses = GlobalAnalyses::default();
        this.visit_scope(scope, &analyses, &changes);
    }
}

impl InstructionVisitor for DisaggregateVisitor {
    fn visit_instruction(
        &mut self,
        mut instruction: Instruction,
        _global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(());

        // This needs to run even for aggregates so extract -> construct will be properly replaced
        visitor.visit_operation(&mut instruction.operation, analyses, |_, val| {
            if let Some(replacement) = self.extracted.get(&val.kind).copied() {
                *val = replacement;
            }
        });
        visitor.visit_out(&mut instruction.out, |_, val| {
            if let Some(replacement) = self.extracted.get(&val.kind) {
                *val = *replacement;
            }
        });

        let mut new_instructions = Vec::new();

        match &mut instruction.operation {
            Operation::ConstructAggregate(fields) => {
                let fields = fields.clone();
                self.substitutes.insert(instruction.out().kind, fields);
            }
            Operation::ExtractAggregateField(operands) => {
                let substitutes = self.substitutes.get(&operands.aggregate.kind);
                let substitutes =
                    substitutes.expect("Should have aggregate registered before any extraction");
                let substitute = substitutes[operands.field];
                self.extracted.insert(instruction.out().kind, substitute);
            }
            // Fix validate prints
            Operation::NonSemantic(NonSemantic::Print { format_string, .. }) => {
                for (from, to) in self.extracted.iter() {
                    *format_string = format_string.replace(&format!("{from}"), &format!("{to}"));
                }
                new_instructions.push(instruction);
            }
            _ => new_instructions.push(instruction),
        }

        new_instructions
    }
}
