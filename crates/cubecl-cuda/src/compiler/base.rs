use std::collections::HashSet;

use cubecl_core::{
    ir::{self as gpu, ConstantScalarValue},
    Compiler,
};
use cubecl_runtime::ExecutionMode;

use super::{Instruction, VariableSettings, WarpInstruction};

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, Default)]
pub struct CudaCompiler {
    shared_memories: Vec<super::SharedMemory>,
    local_arrays: Vec<super::LocalArray>,
    rank: bool,
    wrap_size_checked: bool,
    wmma: bool,
    bf16: bool,
    f16: bool,
    shape: bool,
    stride: bool,
    num_inputs: usize,
    num_outputs: usize,
    items: HashSet<super::Item>,
    strategy: ExecutionMode,
    settings: VariableSettings,
}

impl Compiler for CudaCompiler {
    type Representation = super::ComputeKernel;

    fn compile(
        kernel: cubecl_core::ir::KernelDefinition,
        strategy: ExecutionMode,
    ) -> Self::Representation {
        let compiler = Self {
            strategy,
            ..Self::default()
        };
        compiler.compile_shader(kernel)
    }

    fn elem_size(elem: gpu::Elem) -> usize {
        elem.size()
    }

    fn max_shared_memory_size() -> usize {
        // TODO: Find out this value.
        usize::MAX
    }
}

impl CudaCompiler {
    fn compile_shader(mut self, mut value: gpu::KernelDefinition) -> super::ComputeKernel {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let instructions = self.compile_scope(&mut value.body);
        let inputs = value
            .inputs
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let outputs = value
            .outputs
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let named = value
            .named
            .into_iter()
            .map(|(name, binding)| (name, self.compile_binding(binding)))
            .collect();

        let body = super::Body {
            instructions,
            stride: true,
            shape: true,
            shared_memories: self.shared_memories,
            local_arrays: self.local_arrays,
            rank: self.rank,
            wrap_size_checked: self.wrap_size_checked,
            settings: self.settings,
        };

        super::ComputeKernel {
            inputs,
            outputs,
            named,
            cube_dim: value.cube_dim,
            body,
            wmma_activated: self.wmma,
            bf16: self.bf16,
            f16: self.f16,
            items: self.items,
        }
    }

    fn compile_scope(&mut self, scope: &mut gpu::Scope) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        let processing = scope.process();

        for var in processing.variables {
            if let gpu::Variable::Slice { .. } = var {
                continue;
            }
            instructions.push(Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .operations
            .into_iter()
            .for_each(|op| self.compile_operation(&mut instructions, op, scope));

        instructions
    }

    fn compile_operation(
        &mut self,
        instructions: &mut Vec<Instruction>,
        operation: gpu::Operation,
        scope: &mut gpu::Scope,
    ) {
        match operation {
            gpu::Operation::Operator(op) => self.compile_instruction(op, instructions, scope),
            gpu::Operation::Procedure(proc) => self.compile_procedure(instructions, proc, scope),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => match val {
                gpu::Synchronization::SyncUnits => instructions.push(Instruction::SyncThreads),
                gpu::Synchronization::SyncStorage => instructions.push(Instruction::SyncThreads),
            },
            gpu::Operation::Subcube(op) => {
                self.wrap_size_checked = true;
                match op {
                    gpu::Subcube::Sum(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceSum {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Prod(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceProd {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Max(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceMax {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Min(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceMin {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Elect(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::Elect {
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::All(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::All {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Any(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::Any {
                            input: self.compile_variable(op.input),
                            out: self.compile_variable(op.out),
                        }))
                    }
                    gpu::Subcube::Broadcast(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::Broadcast {
                            input: self.compile_variable(op.lhs),
                            id: self.compile_variable(op.rhs),
                            out: self.compile_variable(op.out),
                        }))
                    }
                }
            }
            gpu::Operation::CoopMma(cmma) => instructions.push(self.compile_cmma(cmma)),
        }
    }

    fn compile_cmma(&mut self, cmma: gpu::CoopMma) -> Instruction {
        match cmma {
            gpu::CoopMma::Fill { mat: frag, value } => {
                Instruction::Wmma(super::WmmaInstruction::Fill {
                    frag: self.compile_variable(frag),
                    value: self.compile_variable(value),
                })
            }
            gpu::CoopMma::Load { mat, value, stride } => {
                Instruction::Wmma(super::WmmaInstruction::Load {
                    frag: self.compile_variable(mat),
                    value: self.compile_variable(value),
                    stride: self.compile_variable(stride),
                })
            }
            gpu::CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => Instruction::Wmma(super::WmmaInstruction::Execute {
                frag_a: self.compile_variable(mat_a),
                frag_b: self.compile_variable(mat_b),
                frag_c: self.compile_variable(mat_c),
                frag_d: self.compile_variable(mat_d),
            }),
            gpu::CoopMma::Store {
                output,
                mat,
                stride,
                layout,
            } => Instruction::Wmma(super::WmmaInstruction::Store {
                output: self.compile_variable(output),
                frag: self.compile_variable(mat),
                stride: self.compile_variable(stride),
                layout: self
                    .compile_matrix_layout(layout)
                    .expect("Layout required for store instruction"),
            }),
        }
    }

    fn compile_metadata(&mut self, metadata: gpu::Metadata) -> Instruction {
        match metadata {
            gpu::Metadata::Stride { dim, var, out } => {
                self.stride = true;
                let position = match var {
                    gpu::Variable::GlobalInputArray { id, .. } => id as usize,
                    gpu::Variable::GlobalOutputArray { id, .. } => self.num_inputs + id as usize,
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                Instruction::Stride {
                    dim: self.compile_variable(dim),
                    position,
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Shape { dim, var, out } => {
                self.shape = true;
                let position = match var {
                    gpu::Variable::GlobalInputArray { id, .. } => id as usize,
                    gpu::Variable::GlobalOutputArray { id, .. } => self.num_inputs + id as usize,
                    _ => panic!("Only Input and Output have a shape, got {:?}", var),
                };
                Instruction::Shape {
                    dim: self.compile_variable(dim),
                    position,
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Length { var, out } => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(out);

                match input {
                    super::Variable::Slice { .. } => super::Instruction::SliceLength { input, out },
                    _ => super::Instruction::Length {
                        input,
                        out,
                        num_inputs: self.num_inputs,
                        num_outputs: self.num_outputs,
                    },
                }
            }
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<Instruction>, branch: gpu::Branch) {
        match branch {
            gpu::Branch::If(mut op) => instructions.push(Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            gpu::Branch::IfElse(mut op) => instructions.push(Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            gpu::Branch::Return => instructions.push(Instruction::Return),
            gpu::Branch::Break => instructions.push(Instruction::Break),
            gpu::Branch::RangeLoop(mut range_loop) => instructions.push(Instruction::RangeLoop {
                i: self.compile_variable(range_loop.i),
                start: self.compile_variable(range_loop.start),
                end: self.compile_variable(range_loop.end),
                step: range_loop.step.map(|it| self.compile_variable(it)),
                instructions: self.compile_scope(&mut range_loop.scope),
            }),
            gpu::Branch::Loop(mut op) => instructions.push(Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }
    fn compile_procedure(
        &mut self,
        instructions: &mut Vec<Instruction>,
        proc: gpu::Procedure,
        scope: &mut gpu::Scope,
    ) {
        let mut compile = |scope: &mut gpu::Scope| {
            instructions.extend(self.compile_scope(scope));
        };

        match proc {
            gpu::Procedure::ReadGlobalWithLayout(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::ReadGlobal(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::WriteGlobal(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::ConditionalAssign(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::CheckedIndex(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::CheckedIndexAssign(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::IndexOffsetGlobalWithLayout(proc) => {
                proc.expand(scope);
                compile(scope);
            }
            gpu::Procedure::EarlyReturn(proc) => {
                proc.expand(scope);
                compile(scope);
            }
        }
    }

    fn compile_instruction(
        &mut self,
        value: gpu::Operator,
        instructions: &mut Vec<Instruction>,
        scope: &mut gpu::Scope,
    ) {
        match value {
            gpu::Operator::Add(op) => instructions.push(Instruction::Add(self.compile_binary(op))),
            gpu::Operator::Mul(op) => instructions.push(Instruction::Mul(self.compile_binary(op))),
            gpu::Operator::Div(op) => instructions.push(Instruction::Div(self.compile_binary(op))),
            gpu::Operator::Sub(op) => instructions.push(Instruction::Sub(self.compile_binary(op))),
            gpu::Operator::Assign(op) => {
                instructions.push(Instruction::Assign(self.compile_unary(op)))
            }
            gpu::Operator::Slice(op) => instructions.push(Instruction::Slice {
                input: self.compile_variable(op.input),
                start: self.compile_variable(op.start),
                end: self.compile_variable(op.end),
                out: self.compile_variable(op.out),
            }),
            gpu::Operator::Index(op) => {
                if let ExecutionMode::Checked = self.strategy {
                    if has_length(&op.lhs) {
                        self.compile_procedure(
                            instructions,
                            gpu::Procedure::CheckedIndex(gpu::CheckedIndex {
                                lhs: op.lhs,
                                rhs: op.rhs,
                                out: op.out,
                            }),
                            scope,
                        );

                        return;
                    }
                };

                instructions.push(Instruction::Index(self.compile_binary(op)));
            }
            gpu::Operator::UncheckedIndex(op) => {
                instructions.push(Instruction::Index(self.compile_binary(op)))
            }
            gpu::Operator::IndexAssign(op) => {
                if let ExecutionMode::Checked = self.strategy {
                    if has_length(&op.out) {
                        self.compile_procedure(
                            instructions,
                            gpu::Procedure::CheckedIndexAssign(gpu::CheckedIndexAssign {
                                lhs: op.lhs,
                                rhs: op.rhs,
                                out: op.out,
                            }),
                            scope,
                        );
                        return;
                    }
                };

                instructions.push(Instruction::IndexAssign(self.compile_binary(op)));
            }
            gpu::Operator::UncheckedIndexAssign(op) => {
                instructions.push(Instruction::IndexAssign(self.compile_binary(op)))
            }
            gpu::Operator::Modulo(op) => {
                instructions.push(Instruction::Modulo(self.compile_binary(op)))
            }
            gpu::Operator::Equal(op) => {
                instructions.push(Instruction::Equal(self.compile_binary(op)))
            }
            gpu::Operator::Lower(op) => {
                instructions.push(Instruction::Lower(self.compile_binary(op)))
            }
            gpu::Operator::Greater(op) => {
                instructions.push(Instruction::Greater(self.compile_binary(op)))
            }
            gpu::Operator::LowerEqual(op) => {
                instructions.push(Instruction::LowerEqual(self.compile_binary(op)))
            }
            gpu::Operator::GreaterEqual(op) => {
                instructions.push(Instruction::GreaterEqual(self.compile_binary(op)))
            }
            gpu::Operator::Abs(op) => instructions.push(Instruction::Abs(self.compile_unary(op))),
            gpu::Operator::Exp(op) => instructions.push(Instruction::Exp(self.compile_unary(op))),
            gpu::Operator::Log(op) => instructions.push(Instruction::Log(self.compile_unary(op))),
            gpu::Operator::Log1p(op) => {
                instructions.push(Instruction::Log1p(self.compile_unary(op)))
            }
            gpu::Operator::Cos(op) => instructions.push(Instruction::Cos(self.compile_unary(op))),
            gpu::Operator::Sin(op) => instructions.push(Instruction::Sin(self.compile_unary(op))),
            gpu::Operator::Tanh(op) => instructions.push(Instruction::Tanh(self.compile_unary(op))),
            gpu::Operator::Powf(op) => {
                instructions.push(Instruction::Powf(self.compile_binary(op)))
            }
            gpu::Operator::Sqrt(op) => instructions.push(Instruction::Sqrt(self.compile_unary(op))),
            gpu::Operator::Erf(op) => instructions.push(Instruction::Erf(self.compile_unary(op))),
            gpu::Operator::And(op) => instructions.push(Instruction::And(self.compile_binary(op))),
            gpu::Operator::Or(op) => instructions.push(Instruction::Or(self.compile_binary(op))),
            gpu::Operator::Not(op) => instructions.push(Instruction::Not(self.compile_unary(op))),
            gpu::Operator::Max(op) => instructions.push(Instruction::Max(self.compile_binary(op))),
            gpu::Operator::Min(op) => instructions.push(Instruction::Min(self.compile_binary(op))),
            gpu::Operator::NotEqual(op) => {
                instructions.push(Instruction::NotEqual(self.compile_binary(op)))
            }
            gpu::Operator::BitwiseOr(op) => {
                instructions.push(Instruction::BitwiseOr(self.compile_binary(op)))
            }
            gpu::Operator::BitwiseAnd(op) => {
                instructions.push(Instruction::BitwiseAnd(self.compile_binary(op)))
            }
            gpu::Operator::BitwiseXor(op) => {
                instructions.push(Instruction::BitwiseXor(self.compile_binary(op)))
            }
            gpu::Operator::ShiftLeft(op) => {
                instructions.push(Instruction::ShiftLeft(self.compile_binary(op)))
            }
            gpu::Operator::ShiftRight(op) => {
                instructions.push(Instruction::ShiftRight(self.compile_binary(op)))
            }
            gpu::Operator::Clamp(op) => instructions.push(Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(op.out),
            }),
            gpu::Operator::Recip(op) => {
                let elem = op.input.item().elem();
                let lhs = match elem {
                    gpu::Elem::Float(kind) => ConstantScalarValue::Float(1.0, kind),
                    gpu::Elem::Int(kind) => ConstantScalarValue::Int(1, kind),
                    gpu::Elem::UInt => ConstantScalarValue::UInt(1),
                    gpu::Elem::Bool => ConstantScalarValue::Bool(true),
                    gpu::Elem::AtomicInt(_) | gpu::Elem::AtomicUInt => {
                        panic!("Cannot use recip with atomics")
                    }
                };

                instructions.push(Instruction::Div(super::BinaryInstruction {
                    lhs: super::Variable::ConstantScalar(lhs, self.compile_elem(elem)),
                    rhs: self.compile_variable(op.input),
                    out: self.compile_variable(op.out),
                }))
            }
            gpu::Operator::Round(op) => {
                instructions.push(Instruction::Round(self.compile_unary(op)))
            }
            gpu::Operator::Floor(op) => {
                instructions.push(Instruction::Floor(self.compile_unary(op)))
            }
            gpu::Operator::Ceil(op) => instructions.push(Instruction::Ceil(self.compile_unary(op))),
            gpu::Operator::Remainder(op) => {
                instructions.push(Instruction::Remainder(self.compile_binary(op)))
            }
            gpu::Operator::Fma(op) => instructions.push(Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(op.out),
            }),
            gpu::Operator::Bitcast(op) => {
                instructions.push(Instruction::Bitcast(self.compile_unary(op)))
            }
            gpu::Operator::AtomicLoad(op) => {
                instructions.push(Instruction::AtomicLoad(self.compile_unary(op)))
            }
            gpu::Operator::AtomicStore(op) => {
                instructions.push(Instruction::AtomicStore(self.compile_unary(op)))
            }
            gpu::Operator::AtomicSwap(op) => {
                instructions.push(Instruction::AtomicSwap(self.compile_binary(op)))
            }
            gpu::Operator::AtomicAdd(op) => {
                instructions.push(Instruction::AtomicAdd(self.compile_binary(op)))
            }
            gpu::Operator::AtomicSub(op) => {
                instructions.push(Instruction::AtomicSub(self.compile_binary(op)))
            }
            gpu::Operator::AtomicMax(op) => {
                instructions.push(Instruction::AtomicMax(self.compile_binary(op)))
            }
            gpu::Operator::AtomicMin(op) => {
                instructions.push(Instruction::AtomicMin(self.compile_binary(op)))
            }
            gpu::Operator::AtomicAnd(op) => {
                instructions.push(Instruction::AtomicAnd(self.compile_binary(op)))
            }
            gpu::Operator::AtomicOr(op) => {
                instructions.push(Instruction::AtomicOr(self.compile_binary(op)))
            }
            gpu::Operator::AtomicXor(op) => {
                instructions.push(Instruction::AtomicXor(self.compile_binary(op)))
            }
            gpu::Operator::AtomicCompareAndSwap(op) => instructions.push(Instruction::AtomicCAS {
                input: self.compile_variable(op.input),
                cmp: self.compile_variable(op.cmp),
                val: self.compile_variable(op.val),
                out: self.compile_variable(op.out),
            }),
            gpu::Operator::Normalize(op) => {
                instructions.push(Instruction::Normalize(self.compile_unary(op)))
            }
        };
    }

    fn compile_binary(&mut self, value: gpu::BinaryOperator) -> super::BinaryInstruction {
        super::BinaryInstruction {
            lhs: self.compile_variable(value.lhs),
            rhs: self.compile_variable(value.rhs),
            out: self.compile_variable(value.out),
        }
    }

    fn compile_unary(&mut self, value: gpu::UnaryOperator) -> super::UnaryInstruction {
        super::UnaryInstruction {
            input: self.compile_variable(value.input),
            out: self.compile_variable(value.out),
        }
    }

    fn compile_variable(&mut self, value: gpu::Variable) -> super::Variable {
        match value {
            gpu::Variable::GlobalInputArray { id, item } => {
                super::Variable::GlobalInputArray(id, self.compile_item(item))
            }
            gpu::Variable::GlobalScalar { id, elem } => {
                super::Variable::GlobalScalar(id, self.compile_elem(elem), elem)
            }
            gpu::Variable::Local { id, item, depth } => super::Variable::Local {
                id,
                item: self.compile_item(item),
                depth,
            },
            gpu::Variable::Slice { id, item, depth } => super::Variable::Slice {
                id,
                item: self.compile_item(item),
                depth,
            },
            gpu::Variable::LocalScalar { id, elem, depth } => super::Variable::LocalScalar {
                id,
                elem: self.compile_elem(elem),
                depth,
            },
            gpu::Variable::GlobalOutputArray { id, item } => {
                super::Variable::GlobalOutputArray(id, self.compile_item(item))
            }
            gpu::Variable::ConstantScalar(value) => {
                super::Variable::ConstantScalar(value, self.compile_elem(value.elem()))
            }
            gpu::Variable::SharedMemory { id, item, length } => {
                let item = self.compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == id) {
                    self.shared_memories
                        .push(super::SharedMemory::new(id, item, length));
                }
                super::Variable::SharedMemory(id, item, length)
            }
            gpu::Variable::AbsolutePos => {
                self.settings.idx_global = true;
                super::Variable::IdxGlobal
            }
            gpu::Variable::Rank => {
                self.rank = true;
                super::Variable::Rank
            }
            gpu::Variable::UnitPos => {
                self.settings.thread_idx_global = true;
                super::Variable::ThreadIdxGlobal
            }
            gpu::Variable::UnitPosX => super::Variable::ThreadIdxX,
            gpu::Variable::UnitPosY => super::Variable::ThreadIdxY,
            gpu::Variable::UnitPosZ => super::Variable::ThreadIdxZ,
            gpu::Variable::CubePosX => super::Variable::BlockIdxX,
            gpu::Variable::CubePosY => super::Variable::BlockIdxY,
            gpu::Variable::CubePosZ => super::Variable::BlockIdxZ,
            gpu::Variable::AbsolutePosX => {
                self.settings.absolute_idx.0 = true;
                super::Variable::AbsoluteIdxX
            }
            gpu::Variable::AbsolutePosY => {
                self.settings.absolute_idx.1 = true;
                super::Variable::AbsoluteIdxY
            }
            gpu::Variable::AbsolutePosZ => {
                self.settings.absolute_idx.2 = true;
                super::Variable::AbsoluteIdxZ
            }
            gpu::Variable::CubeDimX => super::Variable::BlockDimX,
            gpu::Variable::CubeDimY => super::Variable::BlockDimY,
            gpu::Variable::CubeDimZ => super::Variable::BlockDimZ,
            gpu::Variable::CubeCountX => super::Variable::GridDimX,
            gpu::Variable::CubeCountY => super::Variable::GridDimY,
            gpu::Variable::CubeCountZ => super::Variable::GridDimZ,
            gpu::Variable::LocalArray {
                id,
                item,
                depth,
                length,
            } => {
                let item = self.compile_item(item);
                if !self
                    .local_arrays
                    .iter()
                    .any(|s| s.index == id && s.depth == depth)
                {
                    self.local_arrays
                        .push(super::LocalArray::new(id, item, depth, length));
                }
                super::Variable::LocalArray(id, item, depth, length)
            }
            gpu::Variable::CubePos => {
                self.settings.block_idx_global = true;
                super::Variable::BlockIdxGlobal
            }
            gpu::Variable::CubeDim => {
                self.settings.block_dim_global = true;
                super::Variable::BlockDimGlobal
            }
            gpu::Variable::CubeCount => {
                self.settings.grid_dim_global = true;
                super::Variable::GridDimGlobal
            }
            gpu::Variable::SubcubeDim => super::Variable::WarpSize,
            gpu::Variable::Matrix { id, mat, depth } => {
                self.wmma = true;
                super::Variable::WmmaFragment {
                    id,
                    frag: self.compile_matrix(mat),
                    depth,
                }
            }
        }
    }

    fn compile_matrix(&mut self, matrix: gpu::Matrix) -> super::Fragment {
        super::Fragment {
            ident: self.compile_matrix_ident(matrix.ident),
            m: matrix.m,
            n: matrix.n,
            k: matrix.k,
            elem: self.compile_elem(matrix.elem),
            layout: self.compile_matrix_layout(matrix.layout),
        }
    }

    fn compile_matrix_ident(&mut self, ident: gpu::MatrixIdent) -> super::FragmentIdent {
        match ident {
            gpu::MatrixIdent::A => super::FragmentIdent::A,
            gpu::MatrixIdent::B => super::FragmentIdent::B,
            gpu::MatrixIdent::Accumulator => super::FragmentIdent::Accumulator,
        }
    }

    fn compile_matrix_layout(
        &mut self,
        layout: gpu::MatrixLayout,
    ) -> Option<super::FragmentLayout> {
        match layout {
            gpu::MatrixLayout::ColMajor => Some(super::FragmentLayout::ColMajor),
            gpu::MatrixLayout::RowMajor => Some(super::FragmentLayout::RowMajor),
            gpu::MatrixLayout::Undefined => None,
        }
    }

    fn compile_binding(&mut self, binding: gpu::Binding) -> super::Binding {
        super::Binding {
            item: self.compile_item(binding.item),
            size: binding.size,
        }
    }

    fn compile_item(&mut self, item: gpu::Item) -> super::Item {
        let item = super::Item::new(self.compile_elem(item.elem), item.vectorization.into());
        self.items.insert(item);
        self.items.insert(item.optimized());
        item
    }

    fn compile_elem(&mut self, value: gpu::Elem) -> super::Elem {
        match value {
            gpu::Elem::Float(kind) => match kind {
                gpu::FloatKind::F16 => {
                    self.f16 = true;
                    super::Elem::F16
                }
                gpu::FloatKind::BF16 => {
                    self.bf16 = true;
                    super::Elem::BF16
                }
                gpu::FloatKind::F32 => super::Elem::F32,
                gpu::FloatKind::F64 => panic!("f64 isn't supported yet"),
            },
            gpu::Elem::Int(kind) => match kind {
                gpu::IntKind::I32 => super::Elem::I32,
                gpu::IntKind::I64 => panic!("i64 isn't supported yet"),
            },
            gpu::Elem::AtomicInt(kind) => match kind {
                gpu::IntKind::I32 => super::Elem::I32,
                gpu::IntKind::I64 => panic!("atomic<i64> isn't supported yet"),
            },
            gpu::Elem::UInt => super::Elem::U32,
            gpu::Elem::AtomicUInt => super::Elem::U32,
            gpu::Elem::Bool => super::Elem::Bool,
        }
    }
}

fn has_length(var: &gpu::Variable) -> bool {
    matches!(
        var,
        gpu::Variable::GlobalInputArray { .. }
            | gpu::Variable::GlobalOutputArray { .. }
            | gpu::Variable::Slice { .. }
    )
}
