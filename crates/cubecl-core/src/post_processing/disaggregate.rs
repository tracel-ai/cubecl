use alloc::{format, vec::Vec};

use cubecl_ir::{
    GlobalState, Instruction, Memory, NonSemantic, Operation, Scope, Value, ValueKind,
};
use hashbrown::HashMap;

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
        global_state: &GlobalState,
        analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(());
        let state = global_state.borrow();
        let allocator = &state.allocator;

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
                let mut fields = fields.clone();
                // Make an immutable copy if the value is mutable
                for field in fields.iter_mut().filter(|it| it.can_mutate()) {
                    if field.is_value() {
                        let new_field = allocator.create_value(field.ty);
                        new_instructions.push(Instruction::new(Operation::Copy(*field), new_field));
                        *field = new_field;
                    } else if !field.is_array() {
                        let new_field = allocator.create_value(field.ty);
                        new_instructions.push(Instruction::new(Memory::Load(*field), new_field));
                        *field = new_field;
                    }
                }
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
