use cubecl_common::ExecutionMode;
use cubecl_ir::{Allocator, ExpandElement, Instruction, Operation, Operator, Processor, Scope};

use crate::{
    io::read_tensor_checked,
    prelude::{FloatExpand, Line, expand_checked_index_assign},
};

#[derive(new)]
pub struct CheckedIoProcessor {
    mode: ExecutionMode,
}

impl Processor for CheckedIoProcessor {
    fn transform(
        &self,
        mut processing: cubecl_ir::ScopeProcessing,
        allocator: Allocator,
    ) -> cubecl_ir::ScopeProcessing {
        if matches!(self.mode, ExecutionMode::Unchecked) {
            return processing;
        }

        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Operator(operator) = &instruction.operation {
                match operator {
                    Operator::Index(op) => {
                        let has_length = op.list.has_length();
                        let is_not_atomic = !op.list.elem().is_atomic();

                        if has_length && is_not_atomic {
                            let list = ExpandElement::Plain(op.list);
                            let index = ExpandElement::Plain(op.index);
                            let mut scope = Scope::root(false).with_allocator(allocator.clone());
                            scope.register_elem::<FloatExpand<0>>(op.list.elem());

                            let input = read_tensor_checked::expand::<Line<FloatExpand<0>>>(
                                &mut scope,
                                list.into(),
                                index.into(),
                            );
                            let tmp_processing = scope.process([]);

                            for inst in tmp_processing.instructions {
                                processing.instructions.push(inst);
                            }
                            for var in tmp_processing.variables {
                                processing.variables.push(var);
                            }

                            processing.instructions.push(Instruction::new(
                                Operation::Copy(*input.expand),
                                instruction.out(),
                            ));
                            continue;
                        }
                    }
                    Operator::IndexAssign(op) => {
                        let out = instruction.out();

                        if out.has_length() {
                            let mut scope = Scope::root(false).with_allocator(allocator.clone());
                            expand_checked_index_assign(&mut scope, op.index, op.value, out);

                            let tmp_processing = scope.process([]);

                            for inst in tmp_processing.instructions {
                                processing.instructions.push(inst);
                            }
                            for var in tmp_processing.variables {
                                processing.variables.push(var);
                            }

                            continue;
                        }
                    }
                    _ => {}
                }
            }

            // When we have nothing to do.
            processing.instructions.push(instruction);
        }
        processing
    }
}
