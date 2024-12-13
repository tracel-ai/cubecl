use std::hash::Hash;
use std::{collections::HashSet, fmt::Debug, num::NonZero};

use cubecl_core::ir::CheckedIndexAssign;
use cubecl_core::{
    ir::{self as gpu},
    prelude::CubePrimitive,
    Compiler, Feature,
};
use cubecl_runtime::{DeviceProperties, ExecutionMode};

use super::{
    AtomicKind, BinaryInstruction, Binding, Body, ComputeKernel, ConstArray, Elem, Fragment,
    FragmentIdent, FragmentLayout, IndexedVariable, Instruction, Item, LocalArray, SharedMemory,
    UnaryInstruction, Variable, VariableSettings, WarpInstruction, WmmaCompiler, WmmaInstruction,
};

pub(super) static COUNTER_TMP_VAR: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

pub trait Dialect:
    WmmaCompiler<Self> + Default + Clone + Copy + Debug + Send + Sync + Eq + Hash + 'static
{
    // includes
    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn include_runtime(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    // types
    fn bfloat16_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn bfloat162_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    // warp instructions (all threads participating)
    fn warp_shuffle(input: &IndexedVariable<Self>, id: &Variable<Self>) -> String;
    fn warp_shuffle_xor(out: &IndexedVariable<Self>) -> String;
    fn warp_shuffle_down(out: &IndexedVariable<Self>) -> String;
    fn warp_all(out: &IndexedVariable<Self>) -> String;
    fn warp_any(out: &IndexedVariable<Self>) -> String;
}

#[derive(Clone, Debug)]
pub struct CompilationOptions {
    pub warp_size: u32,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self { warp_size: 32 }
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, Default)]
pub struct CppCompiler<D: Dialect> {
    shared_memories: Vec<SharedMemory<D>>,
    const_arrays: Vec<ConstArray<D>>,
    local_arrays: Vec<LocalArray<D>>,
    metadata: cubecl_core::Metadata,
    warp_size_checked: bool,
    wmma: bool,
    bf16: bool,
    f16: bool,
    printf: bool,
    num_inputs: usize,
    num_outputs: usize,
    ext_meta_positions: Vec<u32>,
    items: HashSet<Item<D>>,
    strategy: ExecutionMode,
    settings: VariableSettings,
    compilation_options: CompilationOptions,
}

impl<D: Dialect> Compiler for CppCompiler<D> {
    type Representation = ComputeKernel<D>;
    type CompilationOptions = CompilationOptions;

    fn compile(
        kernel: cubecl_core::ir::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        strategy: ExecutionMode,
    ) -> Self::Representation {
        let compiler = Self {
            compilation_options: compilation_options.clone(),
            strategy,
            ..Self::default()
        };
        let ir = compiler.compile_ir(kernel);
        COUNTER_TMP_VAR.store(0, std::sync::atomic::Ordering::Relaxed);
        ir
    }

    fn elem_size(elem: gpu::Elem) -> usize {
        elem.size()
    }

    fn max_shared_memory_size() -> usize {
        49152
    }

    fn local_allocator() -> impl gpu::LocalAllocator {
        gpu::ReusingAllocator::default()
    }
}

impl<D: Dialect> CppCompiler<D> {
    fn compile_ir(mut self, mut value: gpu::KernelDefinition) -> ComputeKernel<D> {
        self.build_metadata(&value);

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

        let body = Body {
            instructions,
            shared_memories: self.shared_memories,
            const_arrays: self.const_arrays,
            local_arrays: self.local_arrays,
            warp_size_checked: self.warp_size_checked,
            settings: self.settings,
        };

        ComputeKernel {
            inputs,
            outputs,
            named,
            cube_dim: value.cube_dim,
            body,
            wmma_activated: self.wmma,
            bf16: self.bf16,
            f16: self.f16,
            items: self.items,
            kernel_name: value.kernel_name,
        }
    }

    fn build_metadata(&mut self, value: &gpu::KernelDefinition) {
        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();

        let mut num_ext = 0;

        for binding in value.inputs.iter().chain(value.outputs.iter()) {
            self.ext_meta_positions.push(num_ext);
            if binding.has_extended_meta {
                num_ext += 1;
            }
        }

        let num_meta = self.num_inputs + self.num_outputs;

        self.metadata = cubecl_core::Metadata::new(num_meta as u32, num_ext);
    }

    pub(crate) fn ext_meta_position(&self, var: gpu::Variable) -> u32 {
        let pos = match var.kind {
            gpu::VariableKind::GlobalInputArray(id) => id as usize,
            gpu::VariableKind::GlobalOutputArray(id) => self.num_inputs + id as usize,
            other => panic!("Only global arrays have metadata, got {other:?}"),
        };
        self.ext_meta_positions[pos]
    }

    fn compile_scope(&mut self, scope: &mut gpu::Scope) -> Vec<Instruction<D>> {
        let mut instructions = Vec::new();

        let const_arrays = scope
            .const_arrays
            .drain(..)
            .map(|(var, values)| ConstArray {
                index: var.index().unwrap(),
                item: self.compile_item(var.item),
                size: values.len() as u32,
                values: values
                    .into_iter()
                    .map(|val| self.compile_variable(val))
                    .collect(),
            })
            .collect::<Vec<_>>();
        self.const_arrays.extend(const_arrays);

        let processing = scope.process();

        for var in processing.variables {
            if let gpu::VariableKind::Slice { .. } = var.kind {
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
        instructions: &mut Vec<Instruction<D>>,
        instruction: gpu::Instruction,
        scope: &mut gpu::Scope,
    ) {
        let out = instruction.out;
        match instruction.operation {
            gpu::Operation::Copy(variable) => {
                instructions.push(Instruction::Assign(UnaryInstruction {
                    input: self.compile_variable(variable),
                    out: self.compile_variable(out.unwrap()),
                }));
            }
            gpu::Operation::Operator(op) => self.compile_instruction(op, out, instructions, scope),
            gpu::Operation::Atomic(op) => self.compile_atomic(op, out, instructions),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => match val {
                gpu::Synchronization::SyncUnits => instructions.push(Instruction::SyncThreads),
                gpu::Synchronization::SyncStorage => instructions.push(Instruction::SyncThreads),
            },
            gpu::Operation::Comment(val) => instructions.push(Instruction::Comment {
                content: val.content,
            }),
            gpu::Operation::Plane(op) => {
                self.warp_size_checked = true;
                let out = self.compile_variable(out.unwrap());
                match op {
                    gpu::Plane::Sum(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceSum {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Prod(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceProd {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Max(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceMax {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Min(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::ReduceMin {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Elect => {
                        instructions.push(Instruction::Wrap(WarpInstruction::Elect { out }))
                    }
                    gpu::Plane::All(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::All {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Any(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::Any {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Broadcast(op) => {
                        instructions.push(Instruction::Wrap(WarpInstruction::Broadcast {
                            input: self.compile_variable(op.lhs),
                            id: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                }
            }
            gpu::Operation::CoopMma(cmma) => instructions.push(self.compile_cmma(cmma, out)),
            gpu::Operation::Debug(debug) => match debug {
                // No good way to attach debug info
                gpu::DebugInfo::BeginCall { .. } | gpu::DebugInfo::EndCall => {}
                gpu::DebugInfo::Print {
                    format_string,
                    args,
                } => {
                    self.printf = true;
                    instructions.push(Instruction::Printf {
                        format_string,
                        args: args
                            .into_iter()
                            .map(|arg| self.compile_variable(arg))
                            .collect(),
                    })
                }
            },
        }
    }

    fn compile_cmma(&mut self, cmma: gpu::CoopMma, out: Option<gpu::Variable>) -> Instruction<D> {
        let out = self.compile_variable(out.unwrap());
        match cmma {
            gpu::CoopMma::Fill { value } => Instruction::Wmma(WmmaInstruction::Fill {
                frag: out,
                value: self.compile_variable(value),
            }),
            gpu::CoopMma::Load {
                value,
                stride,
                layout,
            } => Instruction::Wmma(WmmaInstruction::Load {
                frag: out,
                value: self.compile_variable(value),
                stride: self.compile_variable(stride),
                layout: layout.and_then(|l| self.compile_matrix_layout(l)),
            }),
            gpu::CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => Instruction::Wmma(WmmaInstruction::Execute {
                frag_a: self.compile_variable(mat_a),
                frag_b: self.compile_variable(mat_b),
                frag_c: self.compile_variable(mat_c),
                frag_d: out,
                warp_size: self.compilation_options.warp_size,
            }),
            gpu::CoopMma::Store {
                mat,
                stride,
                layout,
            } => Instruction::Wmma(WmmaInstruction::Store {
                output: out,
                frag: self.compile_variable(mat),
                stride: self.compile_variable(stride),
                layout: self
                    .compile_matrix_layout(layout)
                    .expect("Layout required for store instruction"),
            }),
            gpu::CoopMma::Cast { input } => Instruction::Wmma(WmmaInstruction::Cast {
                input: self.compile_variable(input),
                output: out,
            }),
        }
    }

    fn compile_metadata(
        &mut self,
        metadata: gpu::Metadata,
        out: Option<gpu::Variable>,
    ) -> Instruction<D> {
        let out = out.unwrap();
        match metadata {
            gpu::Metadata::Stride { dim, var } => {
                let position = self.ext_meta_position(var);
                let offset = self.metadata.stride_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Shape { dim, var } => {
                let position = self.ext_meta_position(var);
                let offset = self.metadata.shape_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Rank { var } => {
                let out = self.compile_variable(out);
                let pos = self.ext_meta_position(var);
                let offset = self.metadata.rank_index(pos);
                super::Instruction::Metadata {
                    info_offset: self.compile_variable(offset.into()),
                    out,
                }
            }
            gpu::Metadata::Length { var } => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(out);

                match input {
                    Variable::Slice { .. } => Instruction::SliceLength { input, out },
                    _ => {
                        let id = match input {
                            Variable::GlobalInputArray(id, _) => id as u32,
                            Variable::GlobalOutputArray(id, _) => {
                                self.num_inputs as u32 + id as u32
                            }
                            _ => panic!("Can only get length of global array"),
                        };
                        let offset = self.metadata.len_index(id);
                        Instruction::Metadata {
                            info_offset: self.compile_variable(offset.into()),
                            out,
                        }
                    }
                }
            }
            gpu::Metadata::BufferLength { var } => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(out);

                match input {
                    Variable::Slice { .. } => Instruction::SliceLength { input, out },
                    _ => {
                        let id = match input {
                            Variable::GlobalInputArray(id, _) => id as u32,
                            Variable::GlobalOutputArray(id, _) => {
                                self.num_inputs as u32 + id as u32
                            }
                            _ => panic!("Can only get buffer length of global array"),
                        };
                        let offset = self.metadata.buffer_len_index(id);
                        Instruction::Metadata {
                            info_offset: self.compile_variable(offset.into()),
                            out,
                        }
                    }
                }
            }
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<Instruction<D>>, branch: gpu::Branch) {
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
            gpu::Branch::Switch(mut op) => instructions.push(Instruction::Switch {
                value: self.compile_variable(op.value),
                instructions_default: self.compile_scope(&mut op.scope_default),
                instructions_cases: op
                    .cases
                    .into_iter()
                    .map(|(val, mut block)| {
                        (self.compile_variable(val), self.compile_scope(&mut block))
                    })
                    .collect(),
            }),
            gpu::Branch::Return => instructions.push(Instruction::Return),
            gpu::Branch::Break => instructions.push(Instruction::Break),
            gpu::Branch::RangeLoop(mut range_loop) => instructions.push(Instruction::RangeLoop {
                i: self.compile_variable(range_loop.i),
                start: self.compile_variable(range_loop.start),
                end: self.compile_variable(range_loop.end),
                step: range_loop.step.map(|it| self.compile_variable(it)),
                inclusive: range_loop.inclusive,
                instructions: self.compile_scope(&mut range_loop.scope),
            }),
            gpu::Branch::Loop(mut op) => instructions.push(Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }

    fn compile_atomic(
        &mut self,
        value: gpu::AtomicOp,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::AtomicOp::Load(op) => {
                instructions.push(Instruction::AtomicLoad(self.compile_unary(op, out)))
            }
            gpu::AtomicOp::Store(op) => {
                instructions.push(Instruction::AtomicStore(self.compile_unary(op, out)))
            }
            gpu::AtomicOp::Swap(op) => {
                instructions.push(Instruction::AtomicSwap(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Add(op) => {
                instructions.push(Instruction::AtomicAdd(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Sub(op) => {
                instructions.push(Instruction::AtomicSub(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Max(op) => {
                instructions.push(Instruction::AtomicMax(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Min(op) => {
                instructions.push(Instruction::AtomicMin(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::And(op) => {
                instructions.push(Instruction::AtomicAnd(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Or(op) => {
                instructions.push(Instruction::AtomicOr(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Xor(op) => {
                instructions.push(Instruction::AtomicXor(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::CompareAndSwap(op) => instructions.push(Instruction::AtomicCAS {
                input: self.compile_variable(op.input),
                cmp: self.compile_variable(op.cmp),
                val: self.compile_variable(op.val),
                out: self.compile_variable(out),
            }),
        }
    }

    fn compile_instruction(
        &mut self,
        value: gpu::Operator,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
        scope: &mut gpu::Scope,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Operator::Add(op) => {
                instructions.push(Instruction::Add(self.compile_binary(op, out)))
            }
            gpu::Operator::Mul(op) => {
                instructions.push(Instruction::Mul(self.compile_binary(op, out)))
            }
            gpu::Operator::Div(op) => {
                instructions.push(Instruction::Div(self.compile_binary(op, out)))
            }
            gpu::Operator::Sub(op) => {
                instructions.push(Instruction::Sub(self.compile_binary(op, out)))
            }
            gpu::Operator::Slice(op) => instructions.push(Instruction::Slice {
                input: self.compile_variable(op.input),
                start: self.compile_variable(op.start),
                end: self.compile_variable(op.end),
                out: self.compile_variable(out),
            }),
            gpu::Operator::Index(op) => {
                if matches!(self.strategy, ExecutionMode::Checked) && op.lhs.has_length() {
                    let lhs = op.lhs;
                    let rhs = op.rhs;
                    let array_len = scope.create_local(gpu::Item::new(u32::as_elem()));

                    instructions.extend(self.compile_scope(scope));

                    let length = match lhs.has_buffer_length() {
                        true => gpu::Metadata::BufferLength { var: lhs },
                        false => gpu::Metadata::Length { var: lhs },
                    };

                    instructions.push(self.compile_metadata(length, Some(array_len)));
                    instructions.push(Instruction::CheckedIndex {
                        len: self.compile_variable(array_len),
                        lhs: self.compile_variable(lhs),
                        rhs: self.compile_variable(rhs),
                        out: self.compile_variable(out),
                    });
                } else {
                    instructions.push(Instruction::Index(self.compile_binary(op, out)));
                }
            }
            gpu::Operator::UncheckedIndex(op) => {
                instructions.push(Instruction::Index(self.compile_binary(op, out)))
            }
            gpu::Operator::IndexAssign(op) => {
                if let ExecutionMode::Checked = self.strategy {
                    if out.has_length() {
                        CheckedIndexAssign {
                            lhs: op.lhs,
                            rhs: op.rhs,
                            out,
                        }
                        .expand(scope);
                        instructions.extend(self.compile_scope(scope));
                        return;
                    }
                };

                instructions.push(Instruction::IndexAssign(self.compile_binary(op, out)));
            }
            gpu::Operator::UncheckedIndexAssign(op) => {
                instructions.push(Instruction::IndexAssign(self.compile_binary(op, out)))
            }
            gpu::Operator::Modulo(op) => {
                instructions.push(Instruction::Modulo(self.compile_binary(op, out)))
            }
            gpu::Operator::Equal(op) => {
                instructions.push(Instruction::Equal(self.compile_binary(op, out)))
            }
            gpu::Operator::Lower(op) => {
                instructions.push(Instruction::Lower(self.compile_binary(op, out)))
            }
            gpu::Operator::Greater(op) => {
                instructions.push(Instruction::Greater(self.compile_binary(op, out)))
            }
            gpu::Operator::LowerEqual(op) => {
                instructions.push(Instruction::LowerEqual(self.compile_binary(op, out)))
            }
            gpu::Operator::GreaterEqual(op) => {
                instructions.push(Instruction::GreaterEqual(self.compile_binary(op, out)))
            }
            gpu::Operator::Abs(op) => {
                instructions.push(Instruction::Abs(self.compile_unary(op, out)))
            }
            gpu::Operator::Exp(op) => {
                instructions.push(Instruction::Exp(self.compile_unary(op, out)))
            }
            gpu::Operator::Log(op) => {
                instructions.push(Instruction::Log(self.compile_unary(op, out)))
            }
            gpu::Operator::Log1p(op) => {
                instructions.push(Instruction::Log1p(self.compile_unary(op, out)))
            }
            gpu::Operator::Cos(op) => {
                instructions.push(Instruction::Cos(self.compile_unary(op, out)))
            }
            gpu::Operator::Sin(op) => {
                instructions.push(Instruction::Sin(self.compile_unary(op, out)))
            }
            gpu::Operator::Tanh(op) => {
                instructions.push(Instruction::Tanh(self.compile_unary(op, out)))
            }
            gpu::Operator::Powf(op) => {
                instructions.push(Instruction::Powf(self.compile_binary(op, out)))
            }
            gpu::Operator::Sqrt(op) => {
                instructions.push(Instruction::Sqrt(self.compile_unary(op, out)))
            }
            gpu::Operator::Erf(op) => {
                instructions.push(Instruction::Erf(self.compile_unary(op, out)))
            }
            gpu::Operator::And(op) => {
                instructions.push(Instruction::And(self.compile_binary(op, out)))
            }
            gpu::Operator::Or(op) => {
                instructions.push(Instruction::Or(self.compile_binary(op, out)))
            }
            gpu::Operator::Not(op) => {
                instructions.push(Instruction::Not(self.compile_unary(op, out)))
            }
            gpu::Operator::Max(op) => {
                instructions.push(Instruction::Max(self.compile_binary(op, out)))
            }
            gpu::Operator::Min(op) => {
                instructions.push(Instruction::Min(self.compile_binary(op, out)))
            }
            gpu::Operator::NotEqual(op) => {
                instructions.push(Instruction::NotEqual(self.compile_binary(op, out)))
            }
            gpu::Operator::BitwiseOr(op) => {
                instructions.push(Instruction::BitwiseOr(self.compile_binary(op, out)))
            }
            gpu::Operator::BitwiseAnd(op) => {
                instructions.push(Instruction::BitwiseAnd(self.compile_binary(op, out)))
            }
            gpu::Operator::BitwiseXor(op) => {
                instructions.push(Instruction::BitwiseXor(self.compile_binary(op, out)))
            }
            gpu::Operator::ShiftLeft(op) => {
                instructions.push(Instruction::ShiftLeft(self.compile_binary(op, out)))
            }
            gpu::Operator::ShiftRight(op) => {
                instructions.push(Instruction::ShiftRight(self.compile_binary(op, out)))
            }
            gpu::Operator::Clamp(op) => instructions.push(Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(out),
            }),
            gpu::Operator::Recip(op) => {
                let elem = op.input.item.elem();
                let lhs = match elem {
                    gpu::Elem::Float(kind) => gpu::ConstantScalarValue::Float(1.0, kind),
                    gpu::Elem::Int(kind) => gpu::ConstantScalarValue::Int(1, kind),
                    gpu::Elem::UInt(kind) => gpu::ConstantScalarValue::UInt(1, kind),
                    gpu::Elem::Bool => gpu::ConstantScalarValue::Bool(true),
                    gpu::Elem::AtomicInt(_) | gpu::Elem::AtomicUInt(_) => {
                        panic!("Cannot use recip with atomics")
                    }
                };

                instructions.push(Instruction::Div(BinaryInstruction {
                    lhs: Variable::ConstantScalar(lhs, self.compile_elem(elem)),
                    rhs: self.compile_variable(op.input),
                    out: self.compile_variable(out),
                }))
            }
            gpu::Operator::Round(op) => {
                instructions.push(Instruction::Round(self.compile_unary(op, out)))
            }
            gpu::Operator::Floor(op) => {
                instructions.push(Instruction::Floor(self.compile_unary(op, out)))
            }
            gpu::Operator::Ceil(op) => {
                instructions.push(Instruction::Ceil(self.compile_unary(op, out)))
            }
            gpu::Operator::Remainder(op) => {
                instructions.push(Instruction::Remainder(self.compile_binary(op, out)))
            }
            gpu::Operator::Fma(op) => instructions.push(Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(out),
            }),
            gpu::Operator::Bitcast(op) => {
                instructions.push(Instruction::Bitcast(self.compile_unary(op, out)))
            }
            gpu::Operator::Neg(op) => {
                instructions.push(Instruction::Negate(self.compile_unary(op, out)))
            }
            gpu::Operator::Normalize(op) => {
                instructions.push(Instruction::Normalize(self.compile_unary(op, out)))
            }
            gpu::Operator::Magnitude(op) => {
                instructions.push(Instruction::Magnitude(self.compile_unary(op, out)))
            }
            gpu::Operator::Dot(op) => {
                instructions.push(Instruction::Dot(self.compile_binary(op, out)))
            }
            gpu::Operator::InitLine(op) => instructions.push(Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|it| self.compile_variable(it))
                    .collect(),
                out: self.compile_variable(out),
            }),
            gpu::Operator::CopyMemory(op) => instructions.push(Instruction::Copy {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
            }),
            gpu::Operator::CopyMemoryBulk(op) => instructions.push(Instruction::CopyBulk {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
                len: op.len,
            }),
            gpu::Operator::Select(op) => instructions.push(Instruction::Select {
                cond: self.compile_variable(op.cond),
                then: self.compile_variable(op.then),
                or_else: self.compile_variable(op.or_else),
                out: self.compile_variable(out),
            }),
            gpu::Operator::Cast(op) => {
                instructions.push(Instruction::Assign(self.compile_unary(op, out)))
            }
        };
    }

    fn compile_binary(
        &mut self,
        value: gpu::BinaryOperator,
        out: gpu::Variable,
    ) -> BinaryInstruction<D> {
        BinaryInstruction {
            lhs: self.compile_variable(value.lhs),
            rhs: self.compile_variable(value.rhs),
            out: self.compile_variable(out),
        }
    }

    fn compile_unary(
        &mut self,
        value: gpu::UnaryOperator,
        out: gpu::Variable,
    ) -> UnaryInstruction<D> {
        UnaryInstruction {
            input: self.compile_variable(value.input),
            out: self.compile_variable(out),
        }
    }

    fn compile_variable(&mut self, value: gpu::Variable) -> Variable<D> {
        let item = value.item;
        match value.kind {
            gpu::VariableKind::GlobalInputArray(id) => {
                Variable::GlobalInputArray(id, self.compile_item(item))
            }
            gpu::VariableKind::GlobalScalar(id) => {
                Variable::GlobalScalar(id, self.compile_item(item).elem, item.elem)
            }
            gpu::VariableKind::Local { id, depth } => Variable::Local {
                id,
                item: self.compile_item(item),
                depth,
            },
            gpu::VariableKind::Versioned { id, depth, .. } => Variable::Local {
                id,
                item: self.compile_item(item),
                depth,
            },
            gpu::VariableKind::LocalBinding { id, depth } => Variable::ConstLocal {
                id,
                item: self.compile_item(item),
                depth,
            },
            gpu::VariableKind::Slice { id, depth } => Variable::Slice {
                id,
                item: self.compile_item(item),
                depth,
            },
            gpu::VariableKind::GlobalOutputArray(id) => {
                Variable::GlobalOutputArray(id, self.compile_item(item))
            }
            gpu::VariableKind::ConstantScalar(value) => {
                Variable::ConstantScalar(value, self.compile_elem(value.elem()))
            }
            gpu::VariableKind::SharedMemory { id, length } => {
                let item = self.compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == id) {
                    self.shared_memories
                        .push(SharedMemory::new(id, item, length));
                }
                Variable::SharedMemory(id, item, length)
            }
            gpu::VariableKind::ConstantArray { id, length } => {
                let item = self.compile_item(item);
                Variable::ConstantArray(id, item, length)
            }
            gpu::VariableKind::Builtin(builtin) => match builtin {
                gpu::Builtin::AbsolutePos => {
                    self.settings.idx_global = true;
                    Variable::IdxGlobal
                }
                gpu::Builtin::UnitPos => {
                    self.settings.thread_idx_global = true;
                    Variable::ThreadIdxGlobal
                }
                gpu::Builtin::UnitPosX => Variable::ThreadIdxX,
                gpu::Builtin::UnitPosY => Variable::ThreadIdxY,
                gpu::Builtin::UnitPosZ => Variable::ThreadIdxZ,
                gpu::Builtin::CubePosX => Variable::BlockIdxX,
                gpu::Builtin::CubePosY => Variable::BlockIdxY,
                gpu::Builtin::CubePosZ => Variable::BlockIdxZ,
                gpu::Builtin::AbsolutePosX => {
                    self.settings.absolute_idx.0 = true;
                    Variable::AbsoluteIdxX
                }
                gpu::Builtin::AbsolutePosY => {
                    self.settings.absolute_idx.1 = true;
                    Variable::AbsoluteIdxY
                }
                gpu::Builtin::AbsolutePosZ => {
                    self.settings.absolute_idx.2 = true;
                    Variable::AbsoluteIdxZ
                }
                gpu::Builtin::CubeDimX => Variable::BlockDimX,
                gpu::Builtin::CubeDimY => Variable::BlockDimY,
                gpu::Builtin::CubeDimZ => Variable::BlockDimZ,
                gpu::Builtin::CubeCountX => Variable::GridDimX,
                gpu::Builtin::CubeCountY => Variable::GridDimY,
                gpu::Builtin::CubeCountZ => Variable::GridDimZ,
                gpu::Builtin::CubePos => {
                    self.settings.block_idx_global = true;
                    Variable::BlockIdxGlobal
                }
                gpu::Builtin::CubeDim => {
                    self.settings.block_dim_global = true;
                    Variable::BlockDimGlobal
                }
                gpu::Builtin::CubeCount => {
                    self.settings.grid_dim_global = true;
                    Variable::GridDimGlobal
                }
                gpu::Builtin::PlaneDim => Variable::WarpSize,
                gpu::Builtin::UnitPosPlane => Variable::ThreadIdxWarp,
            },
            gpu::VariableKind::LocalArray { id, depth, length } => {
                let item = self.compile_item(item);
                if !self
                    .local_arrays
                    .iter()
                    .any(|s| s.index == id && s.depth == depth)
                {
                    self.local_arrays
                        .push(LocalArray::new(id, item, depth, length));
                }
                Variable::LocalArray(id, item, depth, length)
            }
            gpu::VariableKind::Matrix { id, mat, depth } => {
                self.wmma = true;
                Variable::WmmaFragment {
                    id,
                    frag: self.compile_matrix(mat),
                    depth,
                }
            }
        }
    }

    fn compile_matrix(&mut self, matrix: gpu::Matrix) -> Fragment<D> {
        Fragment {
            ident: self.compile_matrix_ident(matrix.ident),
            m: matrix.m,
            n: matrix.n,
            k: matrix.k,
            elem: self.compile_elem(matrix.elem),
            layout: self.compile_matrix_layout(matrix.layout),
        }
    }

    fn compile_matrix_ident(&mut self, ident: gpu::MatrixIdent) -> FragmentIdent<D> {
        match ident {
            gpu::MatrixIdent::A => FragmentIdent::A,
            gpu::MatrixIdent::B => FragmentIdent::B,
            gpu::MatrixIdent::Accumulator => FragmentIdent::Accumulator,
        }
    }

    fn compile_matrix_layout(&mut self, layout: gpu::MatrixLayout) -> Option<FragmentLayout<D>> {
        match layout {
            gpu::MatrixLayout::ColMajor => Some(FragmentLayout::ColMajor),
            gpu::MatrixLayout::RowMajor => Some(FragmentLayout::RowMajor),
            gpu::MatrixLayout::Undefined => None,
        }
    }

    fn compile_binding(&mut self, binding: gpu::Binding) -> Binding<D> {
        Binding {
            item: self.compile_item(binding.item),
            size: binding.size,
        }
    }

    fn compile_item(&mut self, item: gpu::Item) -> Item<D> {
        let item = Item::new(
            self.compile_elem(item.elem),
            item.vectorization.map(NonZero::get).unwrap_or(1).into(),
        );
        if item.elem != super::Elem::TF32 {
            self.items.insert(item);
            self.items.insert(item.optimized());
        } else {
            // TF32 is represented as `float` in C++
            let mut item = item;
            item.elem = super::Elem::F32;
            self.items.insert(item);
        }

        item
    }

    fn compile_elem(&mut self, value: gpu::Elem) -> Elem<D> {
        match value {
            gpu::Elem::Float(kind) => match kind {
                gpu::FloatKind::F16 => {
                    self.f16 = true;
                    Elem::F16
                }
                gpu::FloatKind::BF16 => {
                    self.bf16 = true;
                    Elem::BF16
                }
                gpu::FloatKind::TF32 => Elem::TF32,
                gpu::FloatKind::Flex32 => Elem::F32,
                gpu::FloatKind::F32 => Elem::F32,
                gpu::FloatKind::F64 => Elem::F64,
            },
            gpu::Elem::Int(kind) => match kind {
                gpu::IntKind::I8 => Elem::I8,
                gpu::IntKind::I16 => Elem::I16,
                gpu::IntKind::I32 => Elem::I32,
                gpu::IntKind::I64 => Elem::I64,
            },
            gpu::Elem::AtomicInt(kind) => match kind {
                gpu::IntKind::I32 => Elem::Atomic(AtomicKind::I32),
                _ => panic!("atomic<{}> isn't supported yet", value),
            },
            gpu::Elem::UInt(kind) => match kind {
                gpu::UIntKind::U8 => Elem::U8,
                gpu::UIntKind::U16 => Elem::U16,
                gpu::UIntKind::U32 => Elem::U32,
                gpu::UIntKind::U64 => Elem::U64,
            },
            gpu::Elem::AtomicUInt(kind) => match kind {
                gpu::UIntKind::U32 => Elem::Atomic(AtomicKind::U32),
                kind => unimplemented!("atomic<{kind:?}> not yet supported"),
            },
            gpu::Elem::Bool => Elem::Bool,
        }
    }
}

pub fn register_supported_types(props: &mut DeviceProperties<Feature>) {
    let supported_types = [
        gpu::Elem::UInt(gpu::UIntKind::U8),
        gpu::Elem::UInt(gpu::UIntKind::U16),
        gpu::Elem::UInt(gpu::UIntKind::U32),
        gpu::Elem::UInt(gpu::UIntKind::U64),
        gpu::Elem::Int(gpu::IntKind::I8),
        gpu::Elem::Int(gpu::IntKind::I16),
        gpu::Elem::Int(gpu::IntKind::I32),
        gpu::Elem::Int(gpu::IntKind::I64),
        gpu::Elem::AtomicInt(gpu::IntKind::I32),
        gpu::Elem::AtomicUInt(gpu::UIntKind::U32),
        gpu::Elem::Float(gpu::FloatKind::BF16),
        gpu::Elem::Float(gpu::FloatKind::F16),
        gpu::Elem::Float(gpu::FloatKind::F32),
        gpu::Elem::Float(gpu::FloatKind::Flex32),
        // Causes CUDA_ERROR_INVALID_VALUE for matmul, disabling until that can be investigated
        //gpu::Elem::Float(gpu::FloatKind::F64),
        gpu::Elem::Bool,
    ];

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}
