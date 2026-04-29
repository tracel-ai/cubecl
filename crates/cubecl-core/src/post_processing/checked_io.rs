use alloc::{string::String, vec::Vec};
use cubecl_ir::{Memory, Operation, Processor, Scope};
use cubecl_runtime::server::ExecutionMode;

use crate::io::*;

#[derive(new, Debug)]
pub struct CheckedIoProcessor {
    mode: ExecutionMode,
    kernel_name: String,
}

impl Processor for CheckedIoProcessor {
    fn transform(&self, processing: cubecl_ir::ScopeProcessing) -> cubecl_ir::ScopeProcessing {
        match self.mode {
            ExecutionMode::Checked => self.transform_checked(processing),
            ExecutionMode::Unchecked => processing,
            ExecutionMode::Validate => self.transform_validate(processing),
        }
    }
}

impl CheckedIoProcessor {
    fn transform_checked(
        &self,
        mut processing: cubecl_ir::ScopeProcessing,
    ) -> cubecl_ir::ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Memory(memory) = &instruction.operation {
                match memory {
                    Memory::Index(op) if op.checked => {
                        let has_length = op.list.has_length();

                        if has_length {
                            let list = op.list;
                            let index = op.index;
                            let scope = Scope::root(false)
                                .with_global_state(processing.global_state.clone());

                            expand_checked_index(
                                &scope,
                                list,
                                index,
                                instruction.out(),
                                op.unroll_factor,
                            );

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
                    Memory::Index(op) if op.checked => {
                        let out = instruction.out();

                        if out.has_length() {
                            let scope = Scope::root(false)
                                .with_global_state(processing.global_state.clone());
                            expand_checked_index_mut(
                                &scope,
                                op.list,
                                op.index,
                                instruction.out(),
                                op.unroll_factor,
                            );

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

    fn transform_validate(
        &self,
        mut processing: cubecl_ir::ScopeProcessing,
    ) -> cubecl_ir::ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Memory(memory) = &instruction.operation {
                match memory {
                    Memory::Index(op) if op.checked => {
                        let has_length = op.list.has_length();

                        if has_length {
                            let list = op.list;
                            let index = op.index;
                            let scope = Scope::root(false)
                                .with_global_state(processing.global_state.clone());

                            expand_validate_index(
                                &scope,
                                list,
                                index,
                                instruction.out(),
                                op.unroll_factor,
                                &self.kernel_name,
                            );

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
                    Memory::Index(op) if op.checked => {
                        let out = instruction.out();

                        if out.has_length() {
                            let scope = Scope::root(false)
                                .with_global_state(processing.global_state.clone());
                            expand_validate_index_mut(
                                &scope,
                                op.list,
                                op.index,
                                instruction.out(),
                                op.unroll_factor,
                                &self.kernel_name,
                            );

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
