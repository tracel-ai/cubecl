use alloc::{string::String, vec::Vec};
use cubecl_ir::{Allocator, Instruction, ManagedVariable, Operation, Operator, Processor, Scope};
use cubecl_runtime::server::ExecutionMode;

use crate::{
    define_scalar, define_size,
    io::{
        expand_checked_index_assign, expand_validate_index_assign, read_tensor_atomic_checked,
        read_tensor_atomic_validate, read_tensor_checked, read_tensor_validate,
    },
    prelude::Vector,
};

define_scalar!(ElemA);
define_size!(SizeA);

#[derive(new, Debug)]
pub struct CheckedIoProcessor {
    mode: ExecutionMode,
    kernel_name: String,
}

impl Processor for CheckedIoProcessor {
    fn transform(
        &self,
        processing: cubecl_ir::ScopeProcessing,
        allocator: Allocator,
    ) -> cubecl_ir::ScopeProcessing {
        match self.mode {
            ExecutionMode::Checked => self.transform_checked(processing, allocator),
            ExecutionMode::Unchecked => processing,
            ExecutionMode::Validate => self.transform_validate(processing, allocator),
        }
    }
}

impl CheckedIoProcessor {
    fn transform_checked(
        &self,
        mut processing: cubecl_ir::ScopeProcessing,
        allocator: Allocator,
    ) -> cubecl_ir::ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Operator(operator) = &instruction.operation {
                match operator {
                    Operator::Index(op) => {
                        let has_length = op.list.has_length();

                        if has_length {
                            let list = ManagedVariable::Plain(op.list);
                            let index = ManagedVariable::Plain(op.index);
                            let mut scope = Scope::root(false)
                                .with_allocator(allocator.clone())
                                .with_types(processing.typemap.clone());
                            scope.register_type::<ElemA>(op.list.storage_type());
                            scope.register_size::<SizeA>(op.list.vector_size());

                            let input = if op.list.ty.is_atomic() {
                                // Atomic can't really be checked, since the pointer needs to be
                                // valid, so the kernel will probably not output the correct value if
                                // not manually checked later, but will at least avoid out-of-bounds
                                // memory access.
                                read_tensor_atomic_checked::expand::<ElemA>(
                                    &mut scope,
                                    list.into(),
                                    index.into(),
                                    op.unroll_factor,
                                )
                                .expand
                            } else {
                                read_tensor_checked::expand::<Vector<ElemA, SizeA>>(
                                    &mut scope,
                                    list.into(),
                                    index.into(),
                                    op.unroll_factor,
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
                            let mut scope = Scope::root(false)
                                .with_allocator(allocator.clone())
                                .with_types(processing.typemap.clone());
                            expand_checked_index_assign(
                                &mut scope,
                                op.index,
                                op.value,
                                out,
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
        allocator: Allocator,
    ) -> cubecl_ir::ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Operator(operator) = &instruction.operation {
                match operator {
                    Operator::Index(op) => {
                        let has_length = op.list.has_length();

                        if has_length {
                            let list = ManagedVariable::Plain(op.list);
                            let index = ManagedVariable::Plain(op.index);
                            let mut scope = Scope::root(false)
                                .with_allocator(allocator.clone())
                                .with_types(processing.typemap.clone());
                            scope.register_type::<ElemA>(op.list.storage_type());
                            scope.register_size::<SizeA>(op.list.vector_size());

                            let input = if op.list.ty.is_atomic() {
                                // Atomic can't really be checked, since the pointer needs to be
                                // valid, so the kernel will probably not output the correct value if
                                // not manually checked later, but will at least avoid out-of-bounds
                                // memory access.
                                read_tensor_atomic_validate::expand::<ElemA>(
                                    &mut scope,
                                    list.into(),
                                    index.into(),
                                    op.unroll_factor,
                                    self.kernel_name.clone(),
                                )
                                .expand
                            } else {
                                read_tensor_validate::expand::<Vector<ElemA, SizeA>>(
                                    &mut scope,
                                    list.into(),
                                    index.into(),
                                    op.unroll_factor,
                                    self.kernel_name.clone(),
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
                            let mut scope = Scope::root(false)
                                .with_allocator(allocator.clone())
                                .with_types(processing.typemap.clone());
                            expand_validate_index_assign(
                                &mut scope,
                                op.index,
                                op.value,
                                out,
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
