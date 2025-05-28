use cubecl_common::ExecutionMode;
use cubecl_ir::{Allocator, ExpandElement, Instruction, Operation, Operator, Processor, Scope};

use crate::{
    io::{read_tensor_atomic_checked, read_tensor_checked},
    prelude::{Line, NumericExpand, expand_checked_index_assign},
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

                        if has_length {
                            let list = ExpandElement::Plain(op.list);
                            let index = ExpandElement::Plain(op.index);
                            let mut scope = Scope::root(false).with_allocator(allocator.clone());
                            scope.register_elem::<NumericExpand<0>>(op.list.elem());

                            let input = if op.list.elem().is_atomic() {
                                // Atomic can't really be checked, since the pointer needs to be
                                // valid, so the kernel will probably not output the correct value if
                                // not manually checked later, but will at least avoid out-of-bounds
                                // memory access.
                                read_tensor_atomic_checked::expand::<NumericExpand<0>>(
                                    &mut scope,
                                    list.into(),
                                    index.into(),
                                )
                                .expand
                            } else {
                                read_tensor_checked::expand::<Line<NumericExpand<0>>>(
                                    &mut scope,
                                    list.into(),
                                    index.into(),
                                )
                                .expand
                            };
                            let tmp_processing = scope.process([]);

                            for inst in tmp_processing.instructions {
                                processing.instructions.push(inst);
                            }
                            for var in tmp_processing.variables {
                                processing.variables.push(var);
                            }

                            processing
                                .instructions
                                .push(Instruction::new(Operation::Copy(*input), instruction.out()));
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
