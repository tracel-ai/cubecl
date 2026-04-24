use alloc::{vec, vec::Vec};
use cubecl_ir::{
    Allocator, Arithmetic, BinaryOperator, Branch, CoopMma, IndexOperator, Instruction,
    MatrixLayout, Memory, Metadata, Operation, OperationReflect, Operator, Processor,
    ScopeProcessing, Variable, VariableKind, VectorSize,
};
use hashbrown::HashMap;

/// The action that should be performed on an instruction, returned by ``IrTransformer::maybe_transform``
pub enum TransformAction {
    /// The transformer doesn't apply to this instruction
    Ignore,
    /// Replace this instruction with one or more other instructions
    Replace(Vec<Instruction>),
}

#[derive(new, Debug)]
pub struct UnrollProcessor {
    max_vector_size: VectorSize,
}

struct Mappings(HashMap<Variable, Vec<Variable>>);

impl Mappings {
    fn get(
        &mut self,
        alloc: &Allocator,
        var: Variable,
        unroll_factor: usize,
        vector_size: VectorSize,
    ) -> Vec<Variable> {
        self.0
            .entry(var)
            .or_insert_with(|| create_unrolled(alloc, &var, vector_size, unroll_factor))
            .to_vec()
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
                    } if value.vector_size() > self.max_vector_size => {
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
                    } if inst.out().vector_size() > self.max_vector_size => {
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
                Operation::TensorIndexing(_) => return TransformAction::Ignore,
                Operation::Branch(_) | Operation::NonSemantic(_) | Operation::Marker(_) => {
                    return TransformAction::Ignore;
                }
                _ => {
                    panic!("Need special handling for unrolling non-reflectable operations")
                }
            }
        }

        let args = inst.operation.args().unwrap_or_default();
        if (inst.out.is_some() && inst.ty().vector_size() > self.max_vector_size)
            || args
                .iter()
                .any(|arg| arg.vector_size() > self.max_vector_size)
        {
            let vector_size = max_vector_size(&inst.out, &args);
            let unroll_factor = vector_size / self.max_vector_size;

            match &inst.operation {
                Operation::Memory(Memory::Index(op)) => {
                    TransformAction::Replace(self.transform_array_index(
                        alloc,
                        inst.out(),
                        op,
                        Memory::Index,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Memory(Memory::IndexMut(op)) => {
                    TransformAction::Replace(self.transform_array_index(
                        alloc,
                        inst.out(),
                        op,
                        Memory::IndexMut,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::ExtractComponent(op)) => {
                    TransformAction::Replace(self.transform_composite_extract(
                        alloc,
                        inst.out(),
                        op,
                        unroll_factor,
                        mappings,
                    ))
                }
                Operation::Operator(Operator::InsertComponent(op)) => TransformAction::Replace(
                    self.transform_composite_insert(alloc, inst.out(), op, unroll_factor, mappings),
                ),
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
        let vector_size = value.vector_size();
        let unroll_factor = vector_size / self.max_vector_size;

        let value = unroll_array(*value, self.max_vector_size, unroll_factor);
        let (mul, offset) = mul_index(alloc, *offset, unroll_factor);
        let load = Instruction::new(
            Operation::CoopMma(CoopMma::Load {
                value,
                stride: *stride,
                offset,
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
        let vector_size = out.vector_size();
        let unroll_factor = vector_size / self.max_vector_size;

        let out = unroll_array(out, self.max_vector_size, unroll_factor);
        let (mul, offset) = mul_index(alloc, *offset, unroll_factor);
        let store = Instruction::new(
            Operation::CoopMma(CoopMma::Store {
                mat: *mat,
                stride: *stride,
                offset,
                layout: *layout,
            }),
            out,
        );
        vec![mul, store]
    }

    /// Transforms indexing into multiple index operations, each offset by 1 from the base. The base
    /// is also multiplied by the unroll factor to compensate for the lower actual vectorization.
    fn transform_array_index(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &IndexOperator,
        operator: impl Fn(IndexOperator) -> Memory,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let (mul, start_idx) = mul_index(alloc, op.index, unroll_factor);
        let mut indices = (0..unroll_factor).map(|i| add_index(alloc, start_idx, i));

        let list = unroll_array(op.list, self.max_vector_size, unroll_factor);

        let out = mappings.get(alloc, out, unroll_factor, self.max_vector_size);
        let mut instructions = vec![mul];
        instructions.extend((0..unroll_factor).flat_map(|i| {
            let (add, idx) = indices.next().unwrap();
            let index = Instruction::new(
                operator(IndexOperator {
                    list,
                    index: idx,
                    vector_size: 0,
                    unroll_factor,
                    checked: op.checked,
                }),
                out[i],
            );
            [add, index]
        }));

        instructions
    }

    /// Transforms a composite index (i.e. `Vector`) that always returns a scalar. Translates the index
    /// to a local index and an unroll index, then indexes the proper variable. Note that this requires
    /// the index to be constant - it needs to be decomposed at compile time, otherwise it wouldn't
    /// work.
    fn transform_composite_extract(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &BinaryOperator,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let index = op
            .rhs
            .as_const()
            .expect("Can't unroll non-constant vector index")
            .as_usize();

        let unroll_idx = index / self.max_vector_size;
        let sub_idx = index % self.max_vector_size;

        let value = mappings.get(alloc, op.lhs, unroll_factor, self.max_vector_size);

        vec![Instruction::new(
            Operator::ExtractComponent(BinaryOperator {
                lhs: value[unroll_idx],
                rhs: sub_idx.into(),
            }),
            out,
        )]
    }

    /// Transforms a composite index assign (i.e. `Vector`) that always takes a scalar. Translates the index
    /// to a local index and an unroll index, then indexes the proper variable. Note that this requires
    /// the index to be constant - it needs to be decomposed at compile time, otherwise it wouldn't
    /// work.
    fn transform_composite_insert(
        &self,
        alloc: &Allocator,
        out: Variable,
        op: &BinaryOperator,
        unroll_factor: usize,
        mappings: &mut Mappings,
    ) -> Vec<Instruction> {
        let index = op
            .lhs
            .as_const()
            .expect("Can't unroll non-constant vector index")
            .as_usize();

        let unroll_idx = index / self.max_vector_size;
        let sub_idx = index % self.max_vector_size;

        let out = mappings.get(alloc, out, unroll_factor, self.max_vector_size);

        vec![Instruction::new(
            Operator::InsertComponent(BinaryOperator {
                lhs: sub_idx.into(),
                rhs: op.rhs,
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
                if var.vector_size() > self.max_vector_size {
                    var.ty = var.ty.with_vector_size(self.max_vector_size);
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
            .map(|out| mappings.get(alloc, out, unroll_factor, self.max_vector_size));
        let args = args
            .into_iter()
            .map(|arg| {
                if arg.vector_size() > 1 {
                    mappings.get(alloc, arg, unroll_factor, self.max_vector_size)
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
                        op.scope.register_all(self.transform_instructions(
                            allocator,
                            op.scope.take_instructions(),
                            mappings,
                        ));
                    }
                    Branch::IfElse(op) => {
                        op.scope_if.register_all(self.transform_instructions(
                            allocator,
                            op.scope_if.take_instructions(),
                            mappings,
                        ));
                        op.scope_else.register_all(self.transform_instructions(
                            allocator,
                            op.scope_else.take_instructions(),
                            mappings,
                        ));
                    }
                    Branch::Switch(op) => {
                        for (_, case) in &mut op.cases {
                            case.register_all(self.transform_instructions(
                                allocator,
                                case.take_instructions(),
                                mappings,
                            ));
                        }
                        op.scope_default.register_all(self.transform_instructions(
                            allocator,
                            op.scope_default.take_instructions(),
                            mappings,
                        ));
                    }
                    Branch::RangeLoop(op) => {
                        op.scope.register_all(self.transform_instructions(
                            allocator,
                            op.scope.take_instructions(),
                            mappings,
                        ));
                    }
                    Branch::Loop(op) => {
                        op.scope.register_all(self.transform_instructions(
                            allocator,
                            op.scope.take_instructions(),
                            mappings,
                        ));
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
    fn transform(&self, processing: ScopeProcessing) -> ScopeProcessing {
        let mut mappings = Mappings(Default::default());
        let allocator = processing.global_state.borrow().allocator.clone();

        let instructions =
            self.transform_instructions(&allocator, processing.instructions, &mut mappings);

        ScopeProcessing {
            variables: processing.variables,
            instructions,
            global_state: processing.global_state,
        }
    }
}

fn max_vector_size(out: &Option<Variable>, args: &[Variable]) -> VectorSize {
    let vector_size = args.iter().map(|it| it.vector_size()).max().unwrap();
    vector_size.max(out.map(|out| out.vector_size()).unwrap_or(1))
}

fn create_unrolled(
    allocator: &Allocator,
    var: &Variable,
    max_vector_size: VectorSize,
    unroll_factor: usize,
) -> Vec<Variable> {
    // Preserve scalars
    if var.vector_size() == 1 {
        return vec![*var; unroll_factor];
    }

    let item = var.ty.with_vector_size(max_vector_size);
    (0..unroll_factor)
        .map(|_| match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::Versioned { .. } => {
                allocator.create_local_mut(item)
            }
            VariableKind::Shared { .. } => {
                let id = allocator.new_local_index();
                let shared = VariableKind::Shared { id };
                Variable::new(shared, item)
            }
            VariableKind::LocalConst { .. } => allocator.create_local(item),
            other => panic!("Out must be local, found {other:?}"),
        })
        .collect()
}

fn add_index(alloc: &Allocator, idx: Variable, i: usize) -> (Instruction, Variable) {
    let add_idx = alloc.create_local(idx.ty);
    let add = Instruction::new(
        Arithmetic::Add(BinaryOperator {
            lhs: idx,
            rhs: i.into(),
        }),
        add_idx,
    );
    (add, add_idx)
}

fn mul_index(alloc: &Allocator, idx: Variable, unroll_factor: usize) -> (Instruction, Variable) {
    let mul_idx = alloc.create_local(idx.ty);
    let mul = Instruction::new(
        Arithmetic::Mul(BinaryOperator {
            lhs: idx,
            rhs: unroll_factor.into(),
        }),
        mul_idx,
    );
    (mul, mul_idx)
}

fn unroll_array(mut var: Variable, max_vector_size: VectorSize, factor: usize) -> Variable {
    var.ty = var.ty.with_vector_size(max_vector_size);

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
