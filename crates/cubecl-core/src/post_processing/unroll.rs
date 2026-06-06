use alloc::{vec, vec::Vec};
use cubecl_ir::{
    AddressSpace, Allocator, Arithmetic, BinaryOperands, CoopMma, GlobalState, IndexOperands,
    Instruction, MatrixLayout, Memory, Metadata, Operation, OperationReflect, Operator, Scope,
    Type, Value, ValueKind, VectorInsertOperands, VectorSize,
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
struct Mappings(HashMap<Value, Vec<Value>>);

impl Mappings {
    fn get(
        &mut self,
        alloc: &Allocator,
        val: Value,
        unroll_factor: usize,
        vector_size: VectorSize,
    ) -> Vec<Value> {
        self.0
            .entry(val)
            .or_insert_with(|| create_unrolled(alloc, &val, vector_size, unroll_factor))
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
                Operation::Operator(Operator::ReadBuiltin(_)) => {
                    return TransformAction::Ignore;
                }
                Operation::DeclareVariable {
                    value_ty,
                    addr_space: AddressSpace::Local,
                    alignment,
                } => {
                    if value_ty.vector_size() > self.max_vector_size {
                        let unroll_factor = value_ty.vector_size() / self.max_vector_size;
                        let vector_size = self.max_vector_size;
                        let value_ty = value_ty.with_vector_size(vector_size);
                        let out = self
                            .mappings
                            .get(alloc, inst.out(), unroll_factor, vector_size);
                        let declare = |out| {
                            Instruction::new(
                                Operation::DeclareVariable {
                                    value_ty,
                                    addr_space: AddressSpace::Local,
                                    alignment: *alignment,
                                },
                                out,
                            )
                        };
                        return TransformAction::Replace(out.into_iter().map(declare).collect());
                    } else {
                        return TransformAction::Ignore;
                    }
                }
                Operation::DeclareVariable {
                    value_ty: Type::Array(inner_ty, len),
                    addr_space: AddressSpace::Shared,
                    alignment,
                } => {
                    if inner_ty.vector_size() > self.max_vector_size {
                        let unroll_factor = inner_ty.vector_size() / self.max_vector_size;
                        let vector_size = self.max_vector_size;
                        let inner_ty = inner_ty.with_vector_size(vector_size);
                        let new_ty = Type::Array(inner_ty.intern(), *len * unroll_factor);
                        let new_ptr_ty = Type::Pointer(new_ty.intern(), AddressSpace::Shared);
                        let mut out = inst.out.unwrap();
                        out.ty = new_ptr_ty;

                        return TransformAction::Replace(vec![Instruction::new(
                            Operation::DeclareVariable {
                                value_ty: new_ty,
                                addr_space: AddressSpace::Shared,
                                alignment: *alignment,
                            },
                            out,
                        )]);
                    } else {
                        return TransformAction::Ignore;
                    }
                }
                Operation::DeclareVariable {
                    value_ty,
                    addr_space: AddressSpace::Shared,
                    ..
                } => {
                    if value_ty.vector_size() > self.max_vector_size {
                        todo!()
                    } else {
                        return TransformAction::Ignore;
                    }
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
        out: Value,
        ptr: &Value,
        stride: &Value,
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
        mat: &Value,
        stride: &Value,
        destination: &Value,
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
        out: Value,
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
        out: Value,
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
        out: Value,
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
        (0..unroll_factor)
            .map(|i| {
                if i == unroll_idx {
                    Instruction::new(
                        Operator::InsertComponent(VectorInsertOperands {
                            vector: vector[i],
                            index: sub_idx.into(),
                            value: op.value,
                        }),
                        out[i],
                    )
                } else {
                    Instruction::new(Operation::Copy(vector[i]), out[i])
                }
            })
            .collect()
    }

    /// Transforms metadata by just replacing the type of the buffer. The values are already
    /// properly calculated on the CPU.
    fn transform_metadata(&self, out: Value, op: &Metadata, args: Vec<Value>) -> Vec<Instruction> {
        let op_code = op.op_code();
        let args = args
            .into_iter()
            .map(|mut val| {
                if val.vector_size() > self.max_vector_size {
                    val.ty = val.ty.with_vector_size(self.max_vector_size);
                }
                val
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
        args: Vec<Value>,
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

fn max_vector_size(out: &Option<Value>, args: &[Value]) -> VectorSize {
    let vector_size = args.iter().map(|it| it.vector_size()).max().unwrap();
    vector_size.max(out.map(|out| out.vector_size()).unwrap_or(1))
}

fn create_unrolled(
    allocator: &Allocator,
    val: &Value,
    max_vector_size: VectorSize,
    unroll_factor: usize,
) -> Vec<Value> {
    // Preserve scalars
    if val.vector_size() == 1 {
        return vec![*val; unroll_factor];
    }

    let item = val.ty.with_vector_size(max_vector_size);
    (0..unroll_factor)
        .map(|_| match val.kind {
            ValueKind::Value { .. } => allocator.create_value(item),
            other => panic!("Out must be local, found {other:?}"),
        })
        .collect()
}

fn add_index(alloc: &Allocator, idx: Value, i: usize) -> (Instruction, Value) {
    let add_idx = alloc.create_value(idx.ty);
    let add = Instruction::new(
        Arithmetic::Add(BinaryOperands {
            lhs: idx,
            rhs: i.into(),
        }),
        add_idx,
    );
    (add, add_idx)
}

fn mul_index(alloc: &Allocator, idx: Value, unroll_factor: usize) -> (Instruction, Value) {
    let mul_idx = alloc.create_value(idx.ty);
    let mul = Instruction::new(
        Arithmetic::Mul(BinaryOperands {
            lhs: idx,
            rhs: unroll_factor.into(),
        }),
        mul_idx,
    );
    (mul, mul_idx)
}

fn unroll_array(mut val: Value, max_vector_size: VectorSize, factor: usize) -> Value {
    val.ty = val.ty.with_vector_size(max_vector_size);

    if let Type::Array(_, size) = &mut val.ty {
        *size *= factor;
    }

    val
}
