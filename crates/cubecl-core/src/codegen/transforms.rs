use std::num::NonZero;

use cubecl_ir::{
    Arithmetic, BinaryOperator, CopyMemoryOperator, ExpandElement, IndexAssignOperator,
    IndexOperator, Instruction, Item, Metadata, Operation, OperationReflect, Operator, Scope,
    Variable, VariableKind,
    transformer::{IrTransformer, TransformAction},
};
use hashbrown::HashMap;

use crate::prelude::CubePrimitive;

#[derive(Debug)]
pub struct UnrollTransform {
    max_line_size: u8,
    mappings: HashMap<Variable, Vec<ExpandElement>>,
}

impl UnrollTransform {
    pub fn new(max_line_size: u8) -> Self {
        Self {
            max_line_size,
            mappings: Default::default(),
        }
    }

    fn get_mapping(
        &mut self,
        scope: &mut Scope,
        var: Variable,
        unroll_factor: u8,
    ) -> Vec<Variable> {
        self.mappings
            .entry(var)
            .or_insert_with(|| create_unrolled(scope, &var, self.max_line_size, unroll_factor))
            .iter()
            .map(|it| **it)
            .collect()
    }
}

impl IrTransformer for UnrollTransform {
    fn maybe_transform(&mut self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
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
                    let mut indices_in = vec![op.in_index];
                    let mut indices_out = vec![op.out_index];
                    for _ in 0..unroll_factor as usize - 1 {
                        indices_in
                            .push(*scope.create_local(Item::new(u32::as_elem_native_unchecked())));
                        indices_out
                            .push(*scope.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut input = op.input;
                    input.item.vectorization = NonZero::new(self.max_line_size);

                    let mut out = inst.out();
                    out.item.vectorization = NonZero::new(self.max_line_size);

                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let copy = Instruction::new(
                                Operator::CopyMemory(CopyMemoryOperator {
                                    in_index: indices_in[i],
                                    out_index: indices_out[i],
                                    input,
                                }),
                                out,
                            );
                            if i > 0 {
                                vec![
                                    add_index_inst(op.in_index, i, indices_in[i]),
                                    add_index_inst(op.out_index, i, indices_out[i]),
                                    copy,
                                ]
                            } else {
                                vec![copy]
                            }
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::Index(op)) if op.list.is_array() => {
                    let mut indices = vec![op.index];

                    for _ in 0..unroll_factor as usize - 1 {
                        indices
                            .push(*scope.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut list = op.list;
                    list.item.vectorization = NonZero::new(self.max_line_size);

                    let out = self.get_mapping(scope, inst.out(), unroll_factor);
                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let index = Instruction::new(
                                Operator::Index(IndexOperator {
                                    list,
                                    index: indices[i],
                                    line_size: self.max_line_size as u32,
                                }),
                                out[i],
                            );
                            if i > 0 {
                                vec![add_index_inst(op.index, i, indices[i]), index]
                            } else {
                                vec![index]
                            }
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::UncheckedIndex(op)) if op.list.is_array() => {
                    let mut indices = vec![op.index];

                    for _ in 0..unroll_factor as usize - 1 {
                        indices
                            .push(*scope.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut list = op.list;
                    list.item.vectorization = NonZero::new(self.max_line_size);

                    let out = self.get_mapping(scope, inst.out(), unroll_factor);
                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let index = Instruction::new(
                                Operator::UncheckedIndex(IndexOperator {
                                    list,
                                    index: indices[i],
                                    line_size: self.max_line_size as u32,
                                }),
                                out[i],
                            );
                            if i > 0 {
                                vec![add_index_inst(op.index, i, indices[i]), index]
                            } else {
                                vec![index]
                            }
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

                    let value = self.get_mapping(scope, op.list, unroll_factor);

                    let inst = vec![Instruction::new(
                        Operator::Index(IndexOperator {
                            list: value[unroll_idx as usize],
                            index: sub_idx.into(),
                            line_size: 1,
                        }),
                        inst.out(),
                    )];

                    TransformAction::Replace(inst)
                }
                Operation::Operator(Operator::IndexAssign(op)) if inst.out().is_array() => {
                    let mut indices = vec![op.index];

                    for _ in 0..unroll_factor as usize - 1 {
                        indices
                            .push(*scope.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let mut out = inst.out();
                    out.item.vectorization = NonZero::new(self.max_line_size);

                    let value = self.get_mapping(scope, op.value, unroll_factor);

                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let index = Instruction::new(
                                Operator::IndexAssign(IndexAssignOperator {
                                    index: indices[i],
                                    line_size: self.max_line_size as u32,
                                    value: value[i],
                                }),
                                out,
                            );
                            if i > 0 {
                                vec![add_index_inst(op.index, i, indices[i]), index]
                            } else {
                                vec![index]
                            }
                        })
                        .collect();

                    TransformAction::Replace(instructions)
                }
                Operation::Operator(Operator::UncheckedIndexAssign(op))
                    if inst.out().is_array() =>
                {
                    let mut indices = vec![op.index];

                    for _ in 0..unroll_factor as usize - 1 {
                        indices
                            .push(*scope.create_local(Item::new(u32::as_elem_native_unchecked())));
                    }

                    let value = self.get_mapping(scope, op.value, unroll_factor);

                    let mut out = inst.out();
                    out.item.vectorization = NonZero::new(self.max_line_size);

                    let instructions = (0..unroll_factor as usize)
                        .flat_map(|i| {
                            let index = Instruction::new(
                                Operator::UncheckedIndexAssign(IndexAssignOperator {
                                    index: indices[i],
                                    line_size: self.max_line_size as u32,
                                    value: value[i],
                                }),
                                out,
                            );
                            if i > 0 {
                                vec![add_index_inst(op.index, i, indices[i]), index]
                            } else {
                                vec![index]
                            }
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

                    let out = self.get_mapping(scope, inst.out(), unroll_factor);

                    let inst = vec![Instruction::new(
                        Operator::IndexAssign(IndexAssignOperator {
                            index: sub_idx.into(),
                            line_size: 1,
                            value: op.value,
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
                        .map(|out| self.get_mapping(scope, out, unroll_factor));
                    let args = args
                        .into_iter()
                        .map(|arg| {
                            if arg.vectorization_factor() > 1 {
                                self.get_mapping(scope, arg, unroll_factor)
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
}

fn create_unrolled(
    scope: &mut Scope,
    var: &Variable,
    max_line_size: u8,
    unroll_factor: u8,
) -> Vec<ExpandElement> {
    let item = Item::vectorized(var.elem(), NonZero::new(max_line_size));
    (0..unroll_factor as usize)
        .map(|_| match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::Versioned { .. } => {
                scope.create_local_mut(item)
            }
            VariableKind::LocalConst { .. } => scope.create_local(item),
            _ => panic!("Out must be local"),
        })
        .collect()
}

fn add_index_inst(idx: Variable, i: usize, out: Variable) -> Instruction {
    Instruction::new(
        Arithmetic::Add(BinaryOperator {
            lhs: idx,
            rhs: (i as u32).into(),
        }),
        out,
    )
}
