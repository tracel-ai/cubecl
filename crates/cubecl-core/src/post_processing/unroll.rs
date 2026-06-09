use alloc::{vec, vec::Vec};
use cubecl_ir::{
    Allocator, Arithmetic, BinaryOperands, CoopMma, GlobalState, IndexOperands, Instruction,
    MatrixLayout, Memory, Metadata, Operation, OperationReflect, Operator, Scope, Type, Variable,
    VariableKind, VectorInsertOperands, VectorSize,
};
use hashbrown::HashMap;

use crate::post_processing::{
    analysis_helper::GlobalAnalyses,
    util::AtomicCounter,
    visitor::{InstructionVisitor, visit_scope},
};

/// The action that should be performed on an instruction, returned by ``IrTransformer::maybe_transform``
pub enum TransformAction {
    /// The transformer doesn't apply to this instruction
    Ignore,
    /// Replace this instruction with one or more other instructions
    Replace(Vec<Instruction>),
}

#[derive(Debug)]
pub struct UnrollVisitor {
    max_vector_size: VectorSize,
    mappings: Mappings,
}

#[derive(Default, Debug)]
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

impl UnrollVisitor {
    pub fn new(max_vector_size: VectorSize) -> Self {
        Self {
            max_vector_size,
            mappings: Default::default(),
        }
    }

    pub fn apply(&mut self, scope: &Scope) {
        let changes = AtomicCounter::new(0);
        // We don't care about pointer sources or used variables at this point
        let analyses = GlobalAnalyses::default();
        self.visit_scope(scope, &analyses, &changes);
    }
}

impl InstructionVisitor for UnrollVisitor {
    fn visit_instruction(
        &mut self,
        instruction: Instruction,
        global_state: &GlobalState,
        _analyses: &GlobalAnalyses,
        _changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        match self.maybe_transform(&global_state.borrow().allocator, &instruction) {
            TransformAction::Ignore => {
                vec![instruction]
            }
            TransformAction::Replace(replacement) => replacement,
        }
    }

    fn visit_scope(&mut self, scope: &Scope, analyses: &GlobalAnalyses, changes: &AtomicCounter) {
        visit_scope(self, scope, analyses, changes);

        let state = scope.state();
        let mut locals = state.allocator.local_mut_pool.borrow_mut();

        for variable in locals.iter_mut() {
            if variable.ty.vector_size() > self.max_vector_size {
                let unroll_factor = variable.ty.vector_size() / self.max_vector_size;
                variable.ty = variable.ty.with_vector_size(self.max_vector_size);
                if let Type::Array(_, length, _) = &mut variable.ty {
                    *length *= unroll_factor;
                }
            }
        }
    }
}

impl UnrollVisitor {
    fn maybe_transform(&mut self, alloc: &Allocator, inst: &Instruction) -> TransformAction {
        if matches!(inst.operation, Operation::Marker(_)) {
            return TransformAction::Ignore;
        }

        if inst.operation.args().is_none() {
            // Detect unhandled ops that can't be reflected
            match &inst.operation {
                Operation::CoopMma(op) => match op {
                    // Stride is in scalar elems
                    CoopMma::Load {
                        ptr,
                        stride,
                        layout,
                    } if ptr.vector_size() > self.max_vector_size => {
                        return TransformAction::Replace(self.transform_cmma_load(
                            alloc,
                            inst.out(),
                            ptr,
                            stride,
                            layout,
                        ));
                    }
                    CoopMma::Store {
                        mat,
                        stride,
                        destination,
                        layout,
                    } if destination.vector_size() > self.max_vector_size => {
                        return TransformAction::Replace(self.transform_cmma_store(
                            alloc,
                            mat,
                            stride,
                            destination,
                            layout,
                        ));
                    }
                    _ => return TransformAction::Ignore,
                },
                Operation::TensorIndexing(_) => return TransformAction::Ignore,
                Operation::Branch(_) | Operation::NonSemantic(_) | Operation::Marker(_) => {
                    return TransformAction::Ignore;
                }
                other => {
                    panic!(
                        "Need special handling for unrolling non-reflectable operations.\nFound: {other}"
                    )
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
                Operation::Memory(Memory::Index(op)) => TransformAction::Replace(
                    self.transform_array_index(alloc, inst.out(), op, Memory::Index, unroll_factor),
                ),
                Operation::Operator(Operator::ExtractComponent(op)) => TransformAction::Replace(
                    self.transform_composite_extract(alloc, inst.out(), op, unroll_factor),
                ),
                Operation::Operator(Operator::InsertComponent(op)) => TransformAction::Replace(
                    self.transform_composite_insert(alloc, inst.out(), op, unroll_factor),
                ),
                Operation::Metadata(op) => {
                    TransformAction::Replace(self.transform_metadata(inst.out(), op, args))
                }
                _ => {
                    TransformAction::Replace(self.transform_basic(alloc, inst, args, unroll_factor))
                }
            }
        } else {
            TransformAction::Ignore
        }
    }

    /// Transform CMMA load offset and array
    fn transform_cmma_load(
        &mut self,
        alloc: &Allocator,
        out: Variable,
        ptr: &Variable,
        stride: &Variable,
        layout: &Option<MatrixLayout>,
    ) -> Vec<Instruction> {
        let vector_size = ptr.vector_size();
        let unroll_factor = vector_size / self.max_vector_size;

        let ptr = self
            .mappings
            .get(alloc, *ptr, unroll_factor, self.max_vector_size);
        let out = unroll_array(out, self.max_vector_size, unroll_factor);

        let load = Instruction::new(
            Operation::CoopMma(CoopMma::Load {
                ptr: ptr[0],
                stride: *stride,
                layout: *layout,
            }),
            out,
        );
        vec![load]
    }

    /// Transform CMMA store offset and array
    fn transform_cmma_store(
        &mut self,
        alloc: &Allocator,
        mat: &Variable,
        stride: &Variable,
        destination: &Variable,
        layout: &MatrixLayout,
    ) -> Vec<Instruction> {
        let vector_size = destination.vector_size();
        let unroll_factor = vector_size / self.max_vector_size;

        let destination =
            self.mappings
                .get(alloc, *destination, unroll_factor, self.max_vector_size);

        let store = Instruction::no_out(Operation::CoopMma(CoopMma::Store {
            mat: *mat,
            stride: *stride,
            destination: destination[0],
            layout: *layout,
        }));
        vec![store]
    }

    /// Transforms indexing into multiple index operations, each offset by 1 from the base. The base
    /// is also multiplied by the unroll factor to compensate for the lower actual vectorization.
    fn transform_array_index(
        &mut self,
        alloc: &Allocator,
        out: Variable,
        op: &IndexOperands,
        operator: impl Fn(IndexOperands) -> Memory,
        unroll_factor: usize,
    ) -> Vec<Instruction> {
        let (mul, start_idx) = mul_index(alloc, op.index, unroll_factor);
        let mut indices = (0..unroll_factor).map(|i| add_index(alloc, start_idx, i));

        let list = unroll_array(op.list, self.max_vector_size, unroll_factor);

        let out = self
            .mappings
            .get(alloc, out, unroll_factor, self.max_vector_size);
        let mut instructions = vec![mul];
        instructions.extend((0..unroll_factor).flat_map(|i| {
            let (add, idx) = indices.next().unwrap();
            let index = Instruction::new(
                operator(IndexOperands {
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
        &mut self,
        alloc: &Allocator,
        out: Variable,
        op: &BinaryOperands,
        unroll_factor: usize,
    ) -> Vec<Instruction> {
        let index = op
            .rhs
            .as_const()
            .expect("Can't unroll non-constant vector index")
            .as_usize();

        let unroll_idx = index / self.max_vector_size;
        let sub_idx = index % self.max_vector_size;

        let value = self
            .mappings
            .get(alloc, op.lhs, unroll_factor, self.max_vector_size);

        vec![Instruction::new(
            Operator::ExtractComponent(BinaryOperands {
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
        &mut self,
        alloc: &Allocator,
        out: Variable,
        op: &VectorInsertOperands,
        unroll_factor: usize,
    ) -> Vec<Instruction> {
        let index = op
            .index
            .as_const()
            .expect("Can't unroll non-constant vector index")
            .as_usize();

        let unroll_idx = index / self.max_vector_size;
        let sub_idx = index % self.max_vector_size;

        let vector = self
            .mappings
            .get(alloc, op.vector, unroll_factor, self.max_vector_size);
        let out = self
            .mappings
            .get(alloc, out, unroll_factor, self.max_vector_size);

        vec![Instruction::new(
            Operator::InsertComponent(VectorInsertOperands {
                vector: vector[unroll_idx],
                index: sub_idx.into(),
                value: op.value,
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
        &mut self,
        alloc: &Allocator,
        inst: &Instruction,
        args: Vec<Variable>,
        unroll_factor: usize,
    ) -> Vec<Instruction> {
        let op_code = inst.operation.op_code();
        let out = inst.out.map(|out| {
            self.mappings
                .get(alloc, out, unroll_factor, self.max_vector_size)
        });
        let args = args
            .into_iter()
            .map(|arg| {
                if arg.vector_size() > 1 {
                    self.mappings
                        .get(alloc, arg, unroll_factor, self.max_vector_size)
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
                let shared = VariableKind::Shared {
                    id,
                    alignment: None,
                };
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
        Arithmetic::Add(BinaryOperands {
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
        Arithmetic::Mul(BinaryOperands {
            lhs: idx,
            rhs: unroll_factor.into(),
        }),
        mul_idx,
    );
    (mul, mul_idx)
}

fn unroll_array(mut var: Variable, max_vector_size: VectorSize, factor: usize) -> Variable {
    var.ty = var.ty.with_vector_size(max_vector_size);

    if let VariableKind::ConstantArray { unroll_factor, .. } = &mut var.kind {
        *unroll_factor = factor;
    }

    if let Type::Array(_, size, _) = &mut var.ty {
        *size *= factor;
    }

    var
}
