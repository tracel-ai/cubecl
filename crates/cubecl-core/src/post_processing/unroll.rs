use std::{iter, num::NonZero};

use cubecl_ir::{
    Allocator, Arithmetic, BinaryOperator, Branch, CopyMemoryOperator, ExpandElement,
    IndexAssignOperator, IndexOperator, Instruction, Item, Metadata, Operation, OperationReflect,
    Operator, Processor, ScopeProcessing, Variable, VariableKind,
};
use hashbrown::HashMap;

use crate::prelude::CubePrimitive;

/// The action that should be performed on an instruction, returned by [`IrTransformer::maybe_transform`]
pub enum TransformAction {
    /// The transformer doesn't apply to this instruction
    Ignore,
    /// Replace this instruction with one or more other instructions
    Replace(Vec<Instruction>),
}

#[derive(new, Debug)]
pub struct UnrollProcessor {
    max_line_size: u8,
}

struct Mappings(HashMap<Variable, Vec<ExpandElement>>);

impl Mappings {
    fn get(
        &mut self,
        alloc: &Allocator,
        var: Variable,
        unroll_factor: u8,
        line_size: u8,
    ) -> Vec<Variable> {
        self.0
            .entry(var)
            .or_insert_with(|| create_unrolled(alloc, &var, line_size, unroll_factor))
            .iter()
            .map(|it| **it)
            .collect()
    }
}

impl UnrollProcessor {
    fn maybe_transform(
        &self,
        alloc: &Allocator,
        inst: &Instruction,
        mappings: &mut Mappings,
    ) -> TransformAction {
        if inst.operation.args().is_none() {
            return TransformAction::Ignore;
        }

        let args = inst.operation.args().unwrap_or_default();
        if (inst.out.is_some() && inst.item().vectorization() > self.max_line_size)
            || args
                .iter()
                .any(|arg| arg.vectorization_factor() > self.max_line_size)
        {
            let line_size = args
                .iter()
                .map(|it| it.vectorization_factor())
                .max()
                .unwrap();
            let line_size =
                line_size.max(inst.out.map(|out| out.vectorization_factor()).unwrap_or(1));
            let unroll_factor = line_size / self.max_line_size;

            match &inst.operation {
                Operation::Operator(Operator::CopyMemoryBulk(..)) => TransformAction::Ignore,
                Operation::Operator(Operator::CopyMemory(op)) => {
                    let mut indices_in = vec![];
                    let mut indices_out = vec![];
                    for _ in 0..unroll_factor as usize {
                        indices_in
                            .push(*alloc.create_local(Item::new(u32::as_elem_native_unchecked())));
                        indices_out
                            .push(*alloc.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut input = op.input;
                    input.item.vectorization = NonZero::new(self.max_line_size);

                    let mut out = inst.out();
                    out.item.vectorization = NonZero::new(self.max_line_size);

                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let add_in =
                                add_index_inst(alloc, op.in_index, unroll_factor, i, indices_in[i]);
                            let add_out = add_index_inst(
                                alloc,
                                op.out_index,
                                unroll_factor,
                                i,
                                indices_out[i],
                            );
                            let copy = Instruction::new(
                                Operator::CopyMemory(CopyMemoryOperator {
                                    in_index: indices_in[i],
                                    out_index: indices_out[i],
                                    input,
                                }),
                                out,
                            );
                            add_in.into_iter().chain(add_out).chain(iter::once(copy))
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::Index(op)) if op.list.is_array() => {
                    let mut indices = vec![];

                    for _ in 0..unroll_factor as usize {
                        indices
                            .push(*alloc.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut list = op.list;
                    list.item.vectorization = NonZero::new(self.max_line_size);

                    let out = mappings.get(alloc, inst.out(), unroll_factor, self.max_line_size);
                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let add_idx =
                                add_index_inst(alloc, op.index, unroll_factor, i, indices[i]);
                            let index = Instruction::new(
                                Operator::Index(IndexOperator {
                                    list,
                                    index: indices[i],
                                    line_size: 0,
                                    unroll_factor: unroll_factor as u32,
                                }),
                                out[i],
                            );
                            add_idx.into_iter().chain(iter::once(index))
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::UncheckedIndex(op)) if op.list.is_array() => {
                    let mut indices = vec![];

                    for _ in 0..unroll_factor as usize {
                        indices
                            .push(*alloc.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut list = op.list;
                    list.item.vectorization = NonZero::new(self.max_line_size);

                    let out = mappings.get(alloc, inst.out(), unroll_factor, self.max_line_size);
                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let add_idx =
                                add_index_inst(alloc, op.index, unroll_factor, i, indices[i]);
                            let index = Instruction::new(
                                Operator::UncheckedIndex(IndexOperator {
                                    list,
                                    index: indices[i],
                                    line_size: 0,
                                    unroll_factor: unroll_factor as u32,
                                }),
                                out[i],
                            );
                            add_idx.into_iter().chain(iter::once(index))
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op)) => {
                    let index = op
                        .index
                        .as_const()
                        .expect("Can't unroll non-constant vector index")
                        .as_u32();

                    let unroll_idx = index / self.max_line_size as u32;
                    let sub_idx = index % self.max_line_size as u32;

                    let value = mappings.get(alloc, op.list, unroll_factor, self.max_line_size);

                    let inst = vec![Instruction::new(
                        Operator::Index(IndexOperator {
                            list: value[unroll_idx as usize],
                            index: sub_idx.into(),
                            line_size: 1,
                            unroll_factor: unroll_factor as u32,
                        }),
                        inst.out(),
                    )];

                    TransformAction::Replace(inst)
                }
                Operation::Operator(Operator::IndexAssign(op)) if inst.out().is_array() => {
                    let mut indices = vec![];

                    for _ in 0..unroll_factor as usize {
                        indices
                            .push(*alloc.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut out = inst.out();
                    out.item.vectorization = NonZero::new(self.max_line_size);

                    let value = mappings.get(alloc, op.value, unroll_factor, self.max_line_size);

                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let index = Instruction::new(
                                Operator::IndexAssign(IndexAssignOperator {
                                    index: indices[i],
                                    line_size: 0,
                                    value: value[i],
                                    unroll_factor: unroll_factor as u32,
                                }),
                                out,
                            );
                            let add_idx =
                                add_index_inst(alloc, op.index, unroll_factor, i, indices[i]);

                            add_idx.into_iter().chain(iter::once(index))
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::UncheckedIndexAssign(op))
                    if inst.out().is_array() =>
                {
                    let mut indices = vec![];

                    for _ in 0..unroll_factor as usize {
                        indices
                            .push(*alloc.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let value = mappings.get(alloc, op.value, unroll_factor, self.max_line_size);

                    let mut out = inst.out();
                    out.item.vectorization = NonZero::new(self.max_line_size);

                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let add_idx =
                                add_index_inst(alloc, op.index, unroll_factor, i, indices[i]);
                            let index = Instruction::new(
                                Operator::UncheckedIndexAssign(IndexAssignOperator {
                                    index: indices[i],
                                    line_size: 0,
                                    value: value[i],
                                    unroll_factor: unroll_factor as u32,
                                }),
                                out,
                            );
                            add_idx.into_iter().chain(iter::once(index))
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(
                    Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op),
                ) => {
                    let index = op
                        .index
                        .as_const()
                        .expect("Can't unroll non-constant vector index")
                        .as_u32();

                    let unroll_idx = index / self.max_line_size as u32;
                    let sub_idx = index % self.max_line_size as u32;

                    let out = mappings.get(alloc, inst.out(), unroll_factor, self.max_line_size);

                    let inst = vec![Instruction::new(
                        Operator::IndexAssign(IndexAssignOperator {
                            index: sub_idx.into(),
                            line_size: 1,
                            value: op.value,
                            unroll_factor: unroll_factor as u32,
                        }),
                        out[unroll_idx as usize],
                    )];

                    TransformAction::Replace(inst)
                }
                Operation::Metadata(op) => {
                    let op_code = op.op_code();
                    let args = args
                        .into_iter()
                        .map(|mut var| {
                            if var.vectorization_factor() > self.max_line_size {
                                var.item.vectorization = NonZero::new(self.max_line_size);
                            }
                            var
                        })
                        .collect::<Vec<_>>();
                    let operation = Metadata::from_code_and_args(op_code, &args).unwrap();
                    TransformAction::Replace(vec![Instruction::new(operation, inst.out())])
                }
                op => {
                    let op_code = op.op_code();
                    let out = inst
                        .out
                        .map(|out| mappings.get(alloc, out, unroll_factor, self.max_line_size));
                    let args = args
                        .into_iter()
                        .map(|arg| {
                            if arg.vectorization_factor() > 1 {
                                mappings.get(alloc, arg, unroll_factor, self.max_line_size)
                            } else {
                                // Preserve scalars
                                vec![arg]
                            }
                        })
                        .collect::<Vec<_>>();

                    let instructions = (0..unroll_factor as usize)
                        .map(|i| {
                            let out = out.as_ref().map(|out| out[i]);
                            let args = args
                                .iter()
                                .map(|arg| if arg.len() == 1 { arg[0] } else { arg[i] })
                                .collect::<Vec<_>>();
                            let operation = Operation::from_code_and_args(op_code, &args)
                                .expect("Failed to reconstruct operation");
                            Instruction {
                                out,
                                source_loc: inst.source_loc.clone(),
                                operation,
                            }
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
            }
        } else {
            TransformAction::Ignore
        }
    }

    fn transform_instructions(
        &self,
        allocator: &Allocator,
        instructions: Vec<Instruction>,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let mut new_instructions = Vec::with_capacity(instructions.len());

        for mut instruction in instructions {
            if let Operation::Branch(branch) = &mut instruction.operation {
                match branch {
                    Branch::If(op) => {
                        op.scope.instructions = self.transform_instructions(
                            allocator,
                            op.scope.instructions.drain(..).collect(),
                            mappings,
                        );
                    }
                    Branch::IfElse(op) => {
                        op.scope_if.instructions = self.transform_instructions(
                            allocator,
                            op.scope_if.instructions.drain(..).collect(),
                            mappings,
                        );
                        op.scope_else.instructions = self.transform_instructions(
                            allocator,
                            op.scope_else.instructions.drain(..).collect(),
                            mappings,
                        );
                    }
                    Branch::Switch(op) => {
                        for (_, case) in &mut op.cases {
                            case.instructions = self.transform_instructions(
                                allocator,
                                case.instructions.drain(..).collect(),
                                mappings,
                            );
                        }
                        op.scope_default.instructions = self.transform_instructions(
                            allocator,
                            op.scope_default.instructions.drain(..).collect(),
                            mappings,
                        );
                    }
                    Branch::RangeLoop(op) => {
                        op.scope.instructions = self.transform_instructions(
                            allocator,
                            op.scope.instructions.drain(..).collect(),
                            mappings,
                        );
                    }
                    Branch::Loop(op) => {
                        op.scope.instructions = self.transform_instructions(
                            allocator,
                            op.scope.instructions.drain(..).collect(),
                            mappings,
                        );
                    }
                    _ => {}
                }
            }
            match self.maybe_transform(allocator, &instruction, mappings) {
                TransformAction::Ignore => {
                    new_instructions.push(instruction);
                }
                TransformAction::Replace(replacement) => {
                    new_instructions.extend(replacement);
                }
            }
        }

        new_instructions
    }
}

impl Processor for UnrollProcessor {
    fn transform(&self, processing: ScopeProcessing, allocator: Allocator) -> ScopeProcessing {
        let mut mappings = Mappings(Default::default());

        let instructions =
            self.transform_instructions(&allocator, processing.instructions, &mut mappings);

        ScopeProcessing {
            variables: processing.variables,
            instructions,
        }
    }
}

fn create_unrolled(
    allocator: &Allocator,
    var: &Variable,
    max_line_size: u8,
    unroll_factor: u8,
) -> Vec<ExpandElement> {
    let item = Item::vectorized(var.elem(), NonZero::new(max_line_size));
    (0..unroll_factor as usize)
        .map(|_| match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::Versioned { .. } => {
                allocator.create_local_mut(item)
            }
            VariableKind::LocalConst { .. } => allocator.create_local(item),
            _ => panic!("Out must be local"),
        })
        .collect()
}

fn add_index_inst(
    alloc: &Allocator,
    idx: Variable,
    unroll_factor: u8,
    i: usize,
    out: Variable,
) -> Vec<Instruction> {
    let mul_idx = alloc.create_local(idx.item);
    let mul = Instruction::new(
        Arithmetic::Mul(BinaryOperator {
            lhs: idx,
            rhs: (unroll_factor as u32).into(),
        }),
        *mul_idx,
    );
    let add = Instruction::new(
        Arithmetic::Add(BinaryOperator {
            lhs: *mul_idx,
            rhs: (i as u32).into(),
        }),
        out,
    );
    vec![mul, add]
}
