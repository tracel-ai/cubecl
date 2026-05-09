use alloc::{format, vec::Vec};

use cubecl_ir::{
    GlobalState, Instruction, Memory, NonSemantic, Operation, Scope, Variable, VariableKind,
};
use hashbrown::HashMap;

use crate::post_processing::{
    util::AtomicCounter,
    visitor::{InstructionVisitor, Visitor},
};

type Substitutes = HashMap<VariableKind, Vec<Variable>>;
type Extracted = HashMap<VariableKind, Variable>;

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
        this.visit_scope(scope, &changes);
    }
}

impl InstructionVisitor for DisaggregateVisitor {
    fn visit_instruction(
        &mut self,
        mut instruction: Instruction,
        global_state: &GlobalState,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        let mut visitor = Visitor(());
        let state = global_state.borrow();
        let allocator = &state.allocator;

        // This needs to run even for aggregates so extract -> construct will be properly replaced
        visitor.visit_operation(&mut instruction.operation, |_, var| {
            if let Some(replacement) = self.extracted.get(&var.kind).copied() {
                *var = replacement;
            }
        });
        if let Some(out) = &mut instruction.out
            && let Some(replacement) = self.extracted.get(&out.kind).copied()
        {
            *out = replacement;
        }

        let mut new_instructions = Vec::new();

        match &mut instruction.operation {
            Operation::ConstructAggregate(fields) => {
                let mut fields = fields.clone();
                // Make an immutable copy if the value is mutable
                for field in fields.iter_mut().filter(|it| it.can_mutate()) {
                    if field.is_value() {
                        let new_field = allocator.create_local(field.ty);
                        new_instructions.push(Instruction::new(Operation::Copy(*field), new_field));
                        *field = new_field;
                    } else if !field.is_array() {
                        let new_field = allocator.create_local(field.ty);
                        new_instructions.push(Instruction::new(Memory::Load(*field), new_field));
                        *field = new_field;
                    }
                }
                self.substitutes.insert(instruction.out().kind, fields);
                // std::println!(
                //     "substitutes: {{{}}}",
                //     substitutes
                //         .iter()
                //         .map(|(k, v)| format!("{k}: [{}]", v.iter().join(", ")))
                //         .join(" | ")
                // );
            }
            Operation::ExtractAggregateField(operands) => {
                let substitutes = self.substitutes.get(&operands.aggregate.kind);
                let substitutes =
                    substitutes.expect("Should have aggregate registered before any extraction");
                let substitute = substitutes[operands.field];
                self.extracted.insert(instruction.out().kind, substitute);
                // std::println!(
                //     "extracted: {{{}}}",
                //     extracted
                //         .iter()
                //         .map(|(k, v)| format!("{k}: {v}"))
                //         .join(" | ")
                // );
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
