use cubecl_ir::{
    Allocator, Arithmetic, BinaryOperator, Branch, CoopMma, CopyMemoryBulkOperator, ExpandElement,
    IndexAssignOperator, IndexOperator, Instruction, LineSize, MatrixLayout, Metadata, Operation,
    OperationReflect, Operator, Processor, ScopeProcessing, Type, Variable, VariableKind,
};
use hashbrown::HashMap;

/// The action that should be performed on an instruction, returned by [`IrTransformer::maybe_transform`]
pub enum TransformAction {
    /// The transformer doesn't apply to this instruction
    Ignore,
    /// Replace this instruction with one or more other instructions
    Replace(Vec<Instruction>),
}

#[derive(new, Debug)]
pub struct UnrollProcessor {
    max_line_size: LineSize,
}

struct Mappings(HashMap<Variable, Vec<ExpandElement>>);

impl Mappings {
    fn get(
        &mut self,
        alloc: &Allocator,
        var: Variable,
        unroll_factor: usize,
        line_size: LineSize,
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
        if matches!(inst.operation, Operation::Marker(_)) {
            return TransformAction::Ignore;
        }

        if inst.operation.args().is_none() {
            // Detect unhandled ops that can't be reflected
            match &inst.operation {
                Operation::CoopMma(op) => match op {
                    // Stride is in scalar elems
                    CoopMma::Load {
                        value,
                        stride,
                        offset,
                        layout,
                    } if value.line_size() > self.max_line_size => {
                        return TransformAction::Replace(self.transform_cmma_load(
                            alloc,
                            inst.out(),
                            value,
                            stride,
                            offset,
                            layout,
                        ));
                    }
                    CoopMma::Store {
                        mat,
                        stride,
                        offset,
                        layout,
                    } if inst.out().line_size() > self.max_line_size => {
                        return TransformAction::Replace(self.transform_cmma_store(
                            alloc,
                            inst.out(),
                            mat,
                            stride,
                            offset,
                            layout,
                        ));
                    }
                    _ => return TransformAction::Ignore,
                },
                Operation::Branch(_) | Operation::NonSemantic(_) | Operation::Marker(_) => {
                    return TransformAction::Ignore;
                }
                _ => {
                    panic!("Need special handling for unrolling non-reflectable operations")
                }
            }
        }

        let args = inst.operation.args().unwrap_or_default();
        if (inst.out.is_some() && inst.ty().line_size() > self.max_line_size)
            || args.iter().any(|arg| arg.line_size() > self.max_line_size)
        {
            let line_size = max_line_size(&inst.out, &args);
            let unroll_factor = line_size / self.max_line_size;

            match &inst.operation {
                Operation::Operator(Operator::CopyMemoryBulk(op)) => TransformAction::Replace(
                    self.transform_memcpy(alloc, op, inst.out(), unroll_factor),
                ),
                Operation::Operator(Operator::CopyMemory(op)) => {
                    TransformAction::Replace(self.transform_memcpy(
                        alloc,
                        &CopyMemoryBulkOperator {
                            out_index: op.out_index,
                            input: op.input,
                            in_index: op.in_index,
                            len: 1,
                            offset_input: 0.into(),
                            offset_out: 0.into(),
                        },
                        inst.out(),
                        unroll_factor,
                    ))
                }
                Operation::Operator(Operator::Index(op)) if op.list.is_array() => {
                    TransformAction::Replace(self.transform_array_index(
                        alloc,
                        inst.out(),
                        op,
                        Operator::Index,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::UncheckedIndex(op)) if op.list.is_array() => {
                    TransformAction::Replace(self.transform_array_index(
                        alloc,
                        inst.out(),
                        op,
                        Operator::UncheckedIndex,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::Index(op)) => {
                    TransformAction::Replace(self.transform_composite_index(
                        alloc,
                        inst.out(),
                        op,
                        Operator::Index,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::UncheckedIndex(op)) => {
                    TransformAction::Replace(self.transform_composite_index(
                        alloc,
                        inst.out(),
                        op,
                        Operator::UncheckedIndex,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::IndexAssign(op)) if inst.out().is_array() => {
                    TransformAction::Replace(self.transform_array_index_assign(
                        alloc,
                        inst.out(),
                        op,
                        Operator::IndexAssign,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::UncheckedIndexAssign(op))
                    if inst.out().is_array() =>
                {
                    TransformAction::Replace(self.transform_array_index_assign(
                        alloc,
                        inst.out(),
                        op,
                        Operator::UncheckedIndexAssign,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::IndexAssign(op)) => {
                    TransformAction::Replace(self.transform_composite_index_assign(
                        alloc,
                        inst.out(),
                        op,
                        Operator::IndexAssign,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::UncheckedIndexAssign(op)) => {
                    TransformAction::Replace(self.transform_composite_index_assign(
                        alloc,
                        inst.out(),
                        op,
                        Operator::UncheckedIndexAssign,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Metadata(op) => {
                    TransformAction::Replace(self.transform_metadata(inst.out(), op, args))
                }
                _ => TransformAction::Replace(self.transform_basic(
                    alloc,
                    inst,
                    args,
                    unroll_factor,
                    mappings,
                )),
            }
        } else {
            TransformAction::Ignore
        }
    }

    /// Transform CMMA load offset and array
    fn transform_cmma_load(
        &self,
        alloc: &Allocator,
        out: Variable,
        value: &Variable,
        stride: &Variable,
        offset: &Variable,
        layout: &Option<MatrixLayout>,
    ) -> Vec<Instruction> {
        let line_size = value.line_size();
        let unroll_factor = line_size / self.max_line_size;

        let value = unroll_array(*value, self.max_line_size, unroll_factor);
        let (mul, offset) = mul_index(alloc, *offset, unroll_factor);
        let load = Instruction::new(
            Operation::CoopMma(CoopMma::Load {
                value,
                stride: *stride,
                offset: *offset,
                layout: *layout,
            }),
            out,
        );
        vec![mul, load]
    }

    /// Transform CMMA store offset and array
    fn transform_cmma_store(
        &self,
        alloc: &Allocator,
        out: Variable,
        mat: &Variable,
        stride: &Variable,
        offset: &Variable,
        layout: &MatrixLayout,
    ) -> Vec<Instruction> {
        let line_size = out.line_size();
        let unroll_factor = line_size / self.max_line_size;

        let out = unroll_array(out, self.max_line_size, unroll_factor);
        let (mul, offset) = mul_index(alloc, *offset, unroll_factor);
        let store = Instruction::new(
            Operation::CoopMma(CoopMma::Store {
                mat: *mat,
                stride: *stride,
                offset: *offset,
                layout: *layout,
            }),
            out,
        );
        vec![mul, store]
    }

    /// Transforms memcpy into one with higher length and adjusted indices/offsets
    fn transform_memcpy(
        &self,
        alloc: &Allocator,
        op: &CopyMemoryBulkOperator,
        out: Variable,
        unroll_factor: usize,
    ) -> Vec<Instruction> {
        let (mul1, in_index) = mul_index(alloc, op.in_index, unroll_factor);
        let (mul2, offset_input) = mul_index(alloc, op.offset_input, unroll_factor);
        let (mul3, out_index) = mul_index(alloc, op.out_index, unroll_factor);
        let (mul4, offset_out) = mul_index(alloc, op.offset_out, unroll_factor);

        let input = unroll_array(op.input, self.max_line_size, unroll_factor);
        let out = unroll_array(out, self.max_line_size, unroll_factor);

        vec![
            mul1,
            mul2,
            mul3,
            mul4,
            Instruction::new(
                Operator::CopyMemoryBulk(CopyMemoryBulkOperator {
                    input,
                    in_index: *in_index,
                    out_index: *out_index,
                    len: op.len * unroll_factor,
                    offset_input: *offset_input,
                    offset_out: *offset_out,
                }),
                out,
            ),
        ]
    }

    /// Transforms indexing into multiple index operations, each offset by 1 from the base. The base
    /// is also multiplied by the unroll factor to compensate for the lower actual vectorization.
    fn transform_array_index(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &IndexOperator,
        operator: impl Fn(IndexOperator) -> Operator,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let (mul, start_idx) = mul_index(alloc, op.index, unroll_factor);
        let mut indices = (0..unroll_factor).map(|i| add_index(alloc, *start_idx, i));

        let list = unroll_array(op.list, self.max_line_size, unroll_factor);

        let out = mappings.get(alloc, out, unroll_factor, self.max_line_size);
        let mut instructions = vec![mul];
        instructions.extend((0..unroll_factor).flat_map(|i| {
            let (add, idx) = indices.next().unwrap();
            let index = Instruction::new(
                operator(IndexOperator {
                    list,
                    index: *idx,
                    line_size: 0,
                    unroll_factor,
                }),
                out[i],
            );
            [add, index]
        }));

        instructions
    }

    /// Transforms index assign into multiple index assign operations, each offset by 1 from the base.
    /// The base is also multiplied by the unroll factor to compensate for the lower actual vectorization.
    fn transform_array_index_assign(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &IndexAssignOperator,
        operator: impl Fn(IndexAssignOperator) -> Operator,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let (mul, start_idx) = mul_index(alloc, op.index, unroll_factor);
        let mut indices = (0..unroll_factor).map(|i| add_index(alloc, *start_idx, i));

        let out = unroll_array(out, self.max_line_size, unroll_factor);

        let value = mappings.get(alloc, op.value, unroll_factor, self.max_line_size);

        let mut instructions = vec![mul];
        instructions.extend((0..unroll_factor).flat_map(|i| {
            let (add, idx) = indices.next().unwrap();
            let index = Instruction::new(
                operator(IndexAssignOperator {
                    index: *idx,
                    line_size: 0,
                    value: value[i],
                    unroll_factor,
                }),
                out,
            );

            [add, index]
        }));

        instructions
    }

    /// Transforms a composite index (i.e. `Line`) that always returns a scalar. Translates the index
    /// to a local index and an unroll index, then indexes the proper variable. Note that this requires
    /// the index to be constant - it needs to be decomposed at compile time, otherwise it wouldn't
    /// work.
    fn transform_composite_index(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &IndexOperator,
        operator: impl Fn(IndexOperator) -> Operator,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let index = op
            .index
            .as_const()
            .expect("Can't unroll non-constant vector index")
            .as_usize();

        let unroll_idx = index / self.max_line_size;
        let sub_idx = index % self.max_line_size;

        let value = mappings.get(alloc, op.list, unroll_factor, self.max_line_size);

        vec![Instruction::new(
            operator(IndexOperator {
                list: value[unroll_idx],
                index: sub_idx.into(),
                line_size: 1,
                unroll_factor,
            }),
            out,
        )]
    }

    /// Transforms a composite index assign (i.e. `Line`) that always takes a scalar. Translates the index
    /// to a local index and an unroll index, then indexes the proper variable. Note that this requires
    /// the index to be constant - it needs to be decomposed at compile time, otherwise it wouldn't
    /// work.
    fn transform_composite_index_assign(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &IndexAssignOperator,
        operator: impl Fn(IndexAssignOperator) -> Operator,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let index = op
            .index
            .as_const()
            .expect("Can't unroll non-constant vector index")
            .as_usize();

        let unroll_idx = index / self.max_line_size;
        let sub_idx = index % self.max_line_size;

        let out = mappings.get(alloc, out, unroll_factor, self.max_line_size);

        vec![Instruction::new(
            operator(IndexAssignOperator {
                index: sub_idx.into(),
                line_size: 1,
                value: op.value,
                unroll_factor,
            }),
            out[unroll_idx],
        )]
    }

    /// Transforms metadata by just replacing the type of the buffer. The values are already
    /// properly calculated on the CPU.
    fn transform_metadata(
        &self,
        out: Variable,
        op: &Metadata,
        args: Vec<Variable>,
    ) -> Vec<Instruction> {
        let op_code = op.op_code();
        let args = args
            .into_iter()
            .map(|mut var| {
                if var.line_size() > self.max_line_size {
                    var.ty = var.ty.line(self.max_line_size);
                }
                var
            })
            .collect::<Vec<_>>();
        let operation = Metadata::from_code_and_args(op_code, &args).unwrap();
        vec![Instruction::new(operation, out)]
    }

    /// Transforms generic instructions, i.e. comparison, arithmetic. Unrolls each vectorized variable
    /// to `unroll_factor` replacements, and executes the operation `unroll_factor` times.
    fn transform_basic(
        &self,
        alloc: &Allocator,
        inst: &Instruction,
        args: Vec<Variable>,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let op_code = inst.operation.op_code();
        let out = inst
            .out
            .map(|out| mappings.get(alloc, out, unroll_factor, self.max_line_size));
        let args = args
            .into_iter()
            .map(|arg| {
                if arg.line_size() > 1 {
                    mappings.get(alloc, arg, unroll_factor, self.max_line_size)
                } else {
                    // Preserve scalars
                    vec![arg]
                }
            })
            .collect::<Vec<_>>();

        (0..unroll_factor)
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
                    modes: inst.modes,
                    operation,
                }
            })
            .collect()
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
            typemap: processing.typemap.clone(),
        }
    }
}

fn max_line_size(out: &Option<Variable>, args: &[Variable]) -> LineSize {
    let line_size = args.iter().map(|it| it.line_size()).max().unwrap();
    line_size.max(out.map(|out| out.line_size()).unwrap_or(1))
}

fn create_unrolled(
    allocator: &Allocator,
    var: &Variable,
    max_line_size: LineSize,
    unroll_factor: usize,
) -> Vec<ExpandElement> {
    // Preserve scalars
    if var.line_size() == 1 {
        return vec![ExpandElement::Plain(*var); unroll_factor];
    }

    let item = Type::new(var.storage_type()).line(max_line_size);
    (0..unroll_factor)
        .map(|_| match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::Versioned { .. } => {
                allocator.create_local_mut(item)
            }
            VariableKind::Shared { .. } => {
                let id = allocator.new_local_index();
                let shared = VariableKind::Shared { id };
                ExpandElement::Plain(Variable::new(shared, item))
            }
            VariableKind::LocalConst { .. } => allocator.create_local(item),
            other => panic!("Out must be local, found {other:?}"),
        })
        .collect()
}

fn add_index(alloc: &Allocator, idx: Variable, i: usize) -> (Instruction, ExpandElement) {
    let add_idx = alloc.create_local(idx.ty);
    let add = Instruction::new(
        Arithmetic::Add(BinaryOperator {
            lhs: idx,
            rhs: i.into(),
        }),
        *add_idx,
    );
    (add, add_idx)
}

fn mul_index(
    alloc: &Allocator,
    idx: Variable,
    unroll_factor: usize,
) -> (Instruction, ExpandElement) {
    let mul_idx = alloc.create_local(idx.ty);
    let mul = Instruction::new(
        Arithmetic::Mul(BinaryOperator {
            lhs: idx,
            rhs: unroll_factor.into(),
        }),
        *mul_idx,
    );
    (mul, mul_idx)
}

fn unroll_array(mut var: Variable, max_line_size: LineSize, factor: usize) -> Variable {
    var.ty = var.ty.line(max_line_size);

    match &mut var.kind {
        VariableKind::LocalArray { unroll_factor, .. }
        | VariableKind::ConstantArray { unroll_factor, .. }
        | VariableKind::SharedArray { unroll_factor, .. } => {
            *unroll_factor = factor;
        }
        _ => {}
    }

    var
}
