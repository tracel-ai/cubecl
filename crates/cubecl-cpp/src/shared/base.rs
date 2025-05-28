use std::{collections::HashSet, fmt::Debug, num::NonZero};

use cubecl_common::ExecutionMode;
use cubecl_core::CubeDim;
use cubecl_core::ir::{FloatKind, Processor, UIntKind, VariableKind};
use cubecl_core::post_processing::checked_io::CheckedIoProcessor;
use cubecl_core::{
    Compiler, Feature,
    ir::{self as gpu},
};
use cubecl_core::{
    ir::{Operation, SourceLoc},
    prelude::{FastMath, KernelDefinition},
};
use cubecl_runtime::DeviceProperties;

use super::{
    AtomicKind, BinaryInstruction, Binding, Body, Component, ComputeKernel, ConstArray, Dialect,
    Elem, FP6Kind, Fragment, FragmentIdent, FragmentLayout, IndexAssignInstruction,
    IndexInstruction, Instruction, Item, LocalArray, SharedMemory, UnaryInstruction, Variable,
    WarpInstruction, WmmaInstruction,
};
use super::{FP4Kind, barrier::BarrierOps};
use super::{FP8Kind, pipeline::PipelineOps};

pub(super) static COUNTER_TMP_VAR: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

#[derive(Clone, Debug)]
pub struct CompilationOptions {
    pub warp_size: u32,
    pub grid_constants: bool,
    pub supports_clusters: bool,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            warp_size: 32,
            grid_constants: false,
            supports_clusters: false,
        }
    }
}

/// Cube indexes flags.
/// When true the corresponding index is declared and computed as needed in the kernel.
#[derive(Debug, Clone, Default)]
pub struct CubeIndexFlags {
    pub absolute_pos: bool,
    pub absolute_pos_tuple: bool,
    pub cube_count: bool,
    pub cube_count_tuple: bool,
    pub cube_dim: bool,
    pub cube_dim_tuple: bool,
    pub cube_pos: bool,
    pub cube_pos_tuple: bool,
    pub plane_dim: bool,
    pub plane_dim_checked: bool,
    pub plane_index: bool,
    pub unit_pos: bool,
    pub unit_pos_tuple: bool,
    pub unit_pos_plane: bool,
    pub cluster_pos: bool,
}

/// Flags gathered during Cube IR translation for the kernel compilation.
#[derive(Debug, Clone, Default)]
pub struct Flags {
    pub elem_fp4: bool,
    pub elem_fp6: bool,
    pub elem_fp8: bool,
    pub elem_bf16: bool,
    pub elem_f16: bool,
    pub indexes: CubeIndexFlags,
    pub op_barrier: bool,
    pub op_pipeline: bool,
    pub inst_fast_math: bool,
    pub inst_tma: bool,
    pub inst_tma_im2col: bool,
    pub inst_wmma: bool,
    pub use_grid_constants: bool,
    pub static_meta_length: usize,
    pub has_dynamic_meta: bool,
    pub cluster_dim: Option<CubeDim>,
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, Default)]
pub struct CppCompiler<D: Dialect> {
    barriers: Vec<BarrierOps<D>>,
    compilation_options: CompilationOptions,
    const_arrays: Vec<ConstArray<D>>,
    ext_meta_positions: Vec<u32>,
    cluster_dim: CubeDim,
    extensions: Vec<D::Extension>,
    flags: Flags,
    items: HashSet<Item<D>>,
    local_arrays: Vec<LocalArray<D>>,
    metadata: cubecl_core::Metadata,
    pipelines: Vec<PipelineOps<D>>,
    shared_memories: Vec<SharedMemory<D>>,
    source_loc: Option<SourceLoc>,
    strategy: ExecutionMode,
}

impl<D: Dialect> Compiler for CppCompiler<D> {
    type Representation = ComputeKernel<D>;
    type CompilationOptions = CompilationOptions;

    fn compile(
        &mut self,
        mut kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        strategy: ExecutionMode,
    ) -> Self::Representation {
        self.compilation_options = compilation_options.clone();
        self.strategy = strategy;

        if !self.compilation_options.supports_clusters {
            kernel.options.cluster_dim = None;
        }
        self.cluster_dim = kernel.options.cluster_dim.unwrap_or(CubeDim::new_single());

        let ir = self.clone().compile_ir(kernel);
        COUNTER_TMP_VAR.store(0, std::sync::atomic::Ordering::Relaxed);
        ir
    }

    fn elem_size(&self, elem: gpu::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "cpp"
    }
}

impl<D: Dialect> CppCompiler<D> {
    fn compile_ir(mut self, mut value: KernelDefinition) -> ComputeKernel<D> {
        self.build_metadata(&value);

        let instructions = self.compile_scope(&mut value.body);
        let buffers = value
            .buffers
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let scalars = value
            .scalars
            .into_iter()
            .map(|binding| (self.compile_elem(binding.elem), binding.count))
            .collect();

        // translation flags
        let flags = Flags {
            indexes: D::builtin_rules(&self.flags.indexes),
            inst_wmma: self.flags.inst_wmma,
            op_pipeline: self.flags.op_pipeline,
            op_barrier: self.flags.op_barrier,
            elem_fp4: self.flags.elem_fp4,
            elem_fp6: self.flags.elem_fp6,
            elem_fp8: self.flags.elem_fp8,
            elem_bf16: self.flags.elem_bf16,
            elem_f16: self.flags.elem_f16,
            inst_fast_math: value
                .options
                .fp_math_mode
                .contains(FastMath::ReducedPrecision),
            inst_tma: self.flags.inst_tma,
            inst_tma_im2col: self.flags.inst_tma_im2col,
            use_grid_constants: self.compilation_options.grid_constants,
            // TODO: At some point we should only pass dynamic meta if tensors are present,
            // not if only arrays are present. For now, leave like this
            has_dynamic_meta: self.metadata.static_len() > 0,
            static_meta_length: self.metadata.static_len() as usize,
            cluster_dim: value.options.cluster_dim,
        };

        let body = Body {
            instructions,
            shared_memories: self.shared_memories,
            pipelines: self.pipelines,
            barriers: self.barriers,
            const_arrays: self.const_arrays,
            local_arrays: self.local_arrays,
        };

        let mut cluster_dim = value.options.cluster_dim;
        if !self.compilation_options.supports_clusters {
            cluster_dim = None;
        }

        ComputeKernel {
            tensor_maps: value.tensor_maps,
            buffers,
            scalars,
            meta_static_len: self.metadata.static_len() as usize,
            cube_dim: value.cube_dim,
            body,
            extensions: self.extensions,
            flags,
            items: self.items,
            kernel_name: value.options.kernel_name,
            cluster_dim,
        }
    }

    fn build_metadata(&mut self, value: &KernelDefinition) {
        let mut num_ext = 0;

        let mut all_meta: Vec<_> = value
            .buffers
            .iter()
            .map(|buf| (buf.id, buf.has_extended_meta))
            .chain(value.tensor_maps.iter().map(|i| (*i, true)))
            .collect();

        all_meta.sort_by_key(|(id, _)| *id);

        for (_, has_extended_meta) in &all_meta {
            self.ext_meta_positions.push(num_ext);
            if *has_extended_meta {
                num_ext += 1;
            }
        }

        let num_meta = all_meta.len();

        self.metadata = cubecl_core::Metadata::new(num_meta as u32, num_ext);
    }

    pub(crate) fn ext_meta_position(&self, var: gpu::Variable) -> u32 {
        let id = var.index().expect("Variable should have index");
        self.ext_meta_positions[id as usize]
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

        let checked_io: Box<dyn Processor> = Box::new(CheckedIoProcessor::new(self.strategy));
        let processing = scope.process([checked_io]);

        for var in processing.variables {
            instructions.push(Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .instructions
            .into_iter()
            .for_each(|op| self.compile_instruction(&mut instructions, op));

        instructions
    }

    fn compile_instruction(
        &mut self,
        instructions: &mut Vec<Instruction<D>>,
        instruction: gpu::Instruction,
    ) {
        self.update_debug_loc(instructions, &instruction);
        let out = instruction.out;
        match instruction.operation {
            gpu::Operation::Copy(variable) => {
                instructions.push(Instruction::Assign(UnaryInstruction {
                    input: self.compile_variable(variable),
                    out: self.compile_variable(out.unwrap()),
                }));
            }
            gpu::Operation::Arithmetic(op) => self.compile_arithmetic(op, out, instructions),
            gpu::Operation::Comparison(op) => self.compile_comparison(op, out, instructions),
            gpu::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
            gpu::Operation::Operator(op) => self.compile_operator(op, out, instructions),
            gpu::Operation::Atomic(op) => self.compile_atomic(op, out, instructions),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => match val {
                gpu::Synchronization::SyncCube => instructions.push(Instruction::SyncThreads),
                gpu::Synchronization::SyncPlane => instructions.push(Instruction::SyncWarp),
                gpu::Synchronization::SyncStorage => instructions.push(Instruction::SyncThreads),
                gpu::Synchronization::SyncProxyShared => {
                    self.flags.inst_tma = true;
                    instructions.push(Instruction::ProxySharedFence)
                }
            },
            gpu::Operation::Plane(op) => {
                self.flags.indexes.plane_dim_checked = true;
                let out = self.compile_variable(out.unwrap());
                match op {
                    gpu::Plane::Sum(op) => {
                        let instruction = WarpInstruction::ReduceSum {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction));
                    }
                    gpu::Plane::InclusiveSum(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::InclusiveSum {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::InclusiveProd(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::InclusiveProd {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::ExclusiveSum(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::ExclusiveSum {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::ExclusiveProd(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::ExclusiveProd {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Prod(op) => {
                        let instruction = WarpInstruction::ReduceProd {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    gpu::Plane::Max(op) => {
                        let instruction = WarpInstruction::ReduceMax {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    gpu::Plane::Min(op) => {
                        let instruction = WarpInstruction::ReduceMin {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    gpu::Plane::Elect => {
                        instructions.push(Instruction::Warp(WarpInstruction::Elect { out }))
                    }
                    gpu::Plane::All(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::All {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Any(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Any {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Ballot(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Ballot {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Broadcast(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Broadcast {
                            input: self.compile_variable(op.lhs),
                            id: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                }
            }
            gpu::Operation::CoopMma(cmma) => instructions.push(self.compile_cmma(cmma, out)),
            gpu::Operation::NonSemantic(debug) => match debug {
                gpu::NonSemantic::Print {
                    format_string,
                    args,
                } => instructions.push(Instruction::Printf {
                    format_string,
                    args: args
                        .into_iter()
                        .map(|arg| self.compile_variable(arg))
                        .collect(),
                }),
                gpu::NonSemantic::Comment { content } => {
                    instructions.push(Instruction::Comment { content })
                }
                // Don't need to handle scopes
                _ => {}
            },
            gpu::Operation::Barrier(barrier_ops) => match barrier_ops {
                gpu::BarrierOps::Init {
                    barrier,
                    with_cta_fence,
                } => {
                    let VariableKind::Barrier { level, .. } = barrier.kind else {
                        unreachable!()
                    };
                    let barrier = self.compile_variable(barrier);
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Init {
                        barrier,
                        level,
                        with_cta_fence,
                    }));
                }
                gpu::BarrierOps::MemCopyAsync {
                    barrier,
                    source,
                    source_length,
                    offset_source,
                    offset_out,
                } => {
                    let VariableKind::Barrier { level, .. } = barrier.kind else {
                        unreachable!()
                    };
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsync {
                            barrier: self.compile_variable(barrier),
                            source: self.compile_variable(source),
                            destination: self.compile_variable(out.unwrap()),
                            source_length: self.compile_variable(source_length),
                            offset_source: self.compile_variable(offset_source),
                            offset_out: self.compile_variable(offset_out),
                            level,
                        },
                    ));
                }
                gpu::BarrierOps::TmaLoad {
                    barrier,
                    tensor_map,
                    offset_out,
                    indices,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsyncTensorGlobalToShared {
                            barrier: self.compile_variable(barrier),
                            smem_buffer: self.compile_variable(out.unwrap()),
                            smem_offset: self.compile_variable(offset_out),
                            tensor_map: self.compile_variable(tensor_map),
                            indices: indices
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                        },
                    ));
                }
                gpu::BarrierOps::TmaLoadIm2col {
                    barrier,
                    tensor_map,
                    offset_out,
                    indices,
                    offsets,
                } => {
                    self.flags.inst_tma_im2col = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::TmaLoadIm2col {
                            barrier: self.compile_variable(barrier),
                            smem_buffer: self.compile_variable(out.unwrap()),
                            smem_offset: self.compile_variable(offset_out),
                            tensor_map: self.compile_variable(tensor_map),
                            indices: indices
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                            offsets: offsets
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                        },
                    ));
                }
                gpu::BarrierOps::Arrive { barrier } => {
                    let VariableKind::Barrier { level, .. } = barrier.kind else {
                        unreachable!()
                    };
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Arrive {
                        barrier: self.compile_variable(barrier),
                        level,
                    }))
                }
                gpu::BarrierOps::ArriveTx {
                    barrier,
                    arrive_count_update,
                    transaction_count_update,
                } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::ArriveTx {
                        barrier: self.compile_variable(barrier),
                        arrive_count_update: self.compile_variable(arrive_count_update),
                        transaction_count_update: self.compile_variable(transaction_count_update),
                    }))
                }
                gpu::BarrierOps::ExpectTx {
                    barrier,
                    transaction_count_update,
                } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::ExpectTx {
                        barrier: self.compile_variable(barrier),
                        transaction_count_update: self.compile_variable(transaction_count_update),
                    }))
                }
                gpu::BarrierOps::Wait { barrier } => {
                    let VariableKind::Barrier { level, .. } = barrier.kind else {
                        unreachable!()
                    };
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Wait {
                        barrier: self.compile_variable(barrier),
                        level,
                    }))
                }
                gpu::BarrierOps::ArriveAndWait { barrier } => {
                    let VariableKind::Barrier { level, .. } = barrier.kind else {
                        unreachable!()
                    };
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::ArriveAndWait {
                            barrier: self.compile_variable(barrier),
                            level,
                        },
                    ))
                }
            },
            gpu::Operation::Tma(tma_ops) => {
                self.flags.inst_tma = true;
                match tma_ops {
                    gpu::TmaOps::TmaStore {
                        source,
                        coordinates,
                        offset_source,
                    } => {
                        instructions.push(Instruction::MemCopyAsyncTensorSharedToGlobal {
                            smem_buffer: self.compile_variable(source),
                            smem_offset: self.compile_variable(offset_source),
                            tensor_map: self.compile_variable(out.unwrap()),
                            indices: coordinates
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                        });
                    }
                    gpu::TmaOps::CommitGroup => {
                        instructions.push(Instruction::BulkCommitGroup);
                    }
                    gpu::TmaOps::WaitGroup { max_pending } => {
                        instructions.push(Instruction::BulkWaitGroup { max_pending });
                    }
                    gpu::TmaOps::WaitGroupRead { max_pending } => {
                        instructions.push(Instruction::BulkWaitGroupRead { max_pending });
                    }
                }
            }
        }
    }

    fn update_debug_loc(
        &mut self,
        instructions: &mut Vec<Instruction<D>>,
        inst: &gpu::Instruction,
    ) {
        if !matches!(inst.operation, Operation::NonSemantic(_)) {
            match &inst.source_loc {
                Some(loc) if Some(loc) != self.source_loc.as_ref() => {
                    self.source_loc = Some(loc.clone());
                    instructions.push(Instruction::Line {
                        file: loc.source.file.clone(),
                        line: loc.line,
                    });
                }
                _ => {}
            }
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
                offset,
                layout,
            } => Instruction::Wmma(WmmaInstruction::Load {
                frag: out,
                offset: self.compile_variable(offset),
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
                offset,
                layout,
            } => {
                self.flags.indexes.unit_pos = true;
                self.flags.indexes.plane_index = true;
                Instruction::Wmma(WmmaInstruction::Store {
                    output: out,
                    offset: self.compile_variable(offset),
                    frag: self.compile_variable(mat),
                    stride: self.compile_variable(stride),
                    layout: self
                        .compile_matrix_layout(layout)
                        .expect("Layout required for store instruction"),
                })
            }
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
                    split_meta: self.compilation_options.grid_constants,
                    static_offset: self.metadata.static_len(),
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Shape { dim, var } => {
                let position = self.ext_meta_position(var);
                let offset = self.metadata.shape_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    split_meta: self.compilation_options.grid_constants,
                    static_offset: self.metadata.static_len(),
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Rank { var } => {
                let out = self.compile_variable(out);
                let pos = self.ext_meta_position(var);
                let offset = self.metadata.rank_index(pos);
                super::Instruction::Metadata {
                    info_offset: self.compile_variable(offset.into()),
                    split_meta: self.compilation_options.grid_constants,
                    out,
                }
            }
            gpu::Metadata::Length { var } => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(out);

                match input {
                    Variable::Slice { .. } => Instruction::SliceLength { input, out },
                    Variable::SharedMemory(_id, _item, length) => {
                        Instruction::ConstLength { length, out }
                    }
                    _ => {
                        let id = input.id().expect("Variable should have id");
                        let offset = self.metadata.len_index(id);
                        Instruction::Metadata {
                            info_offset: self.compile_variable(offset.into()),
                            split_meta: self.compilation_options.grid_constants,
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
                        let id = input.id().expect("Variable should have id");
                        let offset = self.metadata.buffer_len_index(id);
                        Instruction::Metadata {
                            info_offset: self.compile_variable(offset.into()),
                            split_meta: self.compilation_options.grid_constants,
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

    fn compile_arithmetic(
        &mut self,
        value: gpu::Arithmetic,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Arithmetic::Add(op) => {
                instructions.push(Instruction::Add(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Mul(op) => {
                instructions.push(Instruction::Mul(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Div(op) => {
                instructions.push(Instruction::Div(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Sub(op) => {
                instructions.push(Instruction::Sub(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::MulHi(op) => {
                let instruction = Instruction::HiMul(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Modulo(op) => {
                instructions.push(Instruction::Modulo(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Abs(op) => {
                instructions.push(Instruction::Abs(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Exp(op) => {
                instructions.push(Instruction::Exp(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Log(op) => {
                instructions.push(Instruction::Log(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Log1p(op) => {
                instructions.push(Instruction::Log1p(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Cos(op) => {
                instructions.push(Instruction::Cos(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Sin(op) => {
                instructions.push(Instruction::Sin(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Tanh(op) => {
                let instruction = Instruction::Tanh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Powf(op) => {
                instructions.push(Instruction::Powf(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Sqrt(op) => {
                instructions.push(Instruction::Sqrt(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Erf(op) => {
                let instruction = Instruction::Erf(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Max(op) => {
                let instruction = Instruction::Max(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Min(op) => {
                let instruction = Instruction::Min(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Clamp(op) => instructions.push(Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(out),
            }),
            gpu::Arithmetic::Recip(op) => {
                let elem = op.input.item.elem();
                let lhs = match elem {
                    gpu::Elem::Float(kind) => gpu::ConstantScalarValue::Float(1.0, kind),
                    gpu::Elem::Int(kind) => gpu::ConstantScalarValue::Int(1, kind),
                    gpu::Elem::UInt(kind) => gpu::ConstantScalarValue::UInt(1, kind),
                    gpu::Elem::Bool => gpu::ConstantScalarValue::Bool(true),
                    gpu::Elem::AtomicInt(_)
                    | gpu::Elem::AtomicUInt(_)
                    | gpu::Elem::AtomicFloat(_) => {
                        panic!("Cannot use recip with atomics")
                    }
                };

                instructions.push(Instruction::Div(BinaryInstruction {
                    lhs: Variable::ConstantScalar(lhs, self.compile_elem(elem)),
                    rhs: self.compile_variable(op.input),
                    out: self.compile_variable(out),
                }))
            }
            gpu::Arithmetic::Round(op) => {
                instructions.push(Instruction::Round(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Floor(op) => {
                instructions.push(Instruction::Floor(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Ceil(op) => {
                instructions.push(Instruction::Ceil(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Remainder(op) => {
                instructions.push(Instruction::Remainder(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Fma(op) => instructions.push(Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(out),
            }),
            gpu::Arithmetic::Neg(op) => {
                instructions.push(Instruction::Neg(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Normalize(op) => {
                instructions.push(Instruction::Normalize(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Magnitude(op) => {
                instructions.push(Instruction::Magnitude(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Dot(op) => {
                instructions.push(Instruction::Dot(self.compile_binary(op, out)))
            }
        };
    }

    fn compile_comparison(
        &mut self,
        value: gpu::Comparison,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Comparison::Equal(op) => {
                instructions.push(Instruction::Equal(self.compile_binary(op, out)))
            }
            gpu::Comparison::Lower(op) => {
                instructions.push(Instruction::Lower(self.compile_binary(op, out)))
            }
            gpu::Comparison::Greater(op) => {
                instructions.push(Instruction::Greater(self.compile_binary(op, out)))
            }
            gpu::Comparison::LowerEqual(op) => {
                instructions.push(Instruction::LowerEqual(self.compile_binary(op, out)))
            }
            gpu::Comparison::GreaterEqual(op) => {
                instructions.push(Instruction::GreaterEqual(self.compile_binary(op, out)))
            }
            gpu::Comparison::NotEqual(op) => {
                instructions.push(Instruction::NotEqual(self.compile_binary(op, out)))
            }
        };
    }

    fn compile_bitwise(
        &mut self,
        value: gpu::Bitwise,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Bitwise::BitwiseOr(op) => {
                instructions.push(Instruction::BitwiseOr(self.compile_binary(op, out)))
            }
            gpu::Bitwise::BitwiseAnd(op) => {
                instructions.push(Instruction::BitwiseAnd(self.compile_binary(op, out)))
            }
            gpu::Bitwise::BitwiseXor(op) => {
                instructions.push(Instruction::BitwiseXor(self.compile_binary(op, out)))
            }
            gpu::Bitwise::CountOnes(op) => {
                instructions.push(Instruction::CountBits(self.compile_unary(op, out)))
            }
            gpu::Bitwise::ReverseBits(op) => {
                instructions.push(Instruction::ReverseBits(self.compile_unary(op, out)))
            }
            gpu::Bitwise::ShiftLeft(op) => {
                instructions.push(Instruction::ShiftLeft(self.compile_binary(op, out)))
            }
            gpu::Bitwise::ShiftRight(op) => {
                instructions.push(Instruction::ShiftRight(self.compile_binary(op, out)))
            }
            gpu::Bitwise::BitwiseNot(op) => {
                instructions.push(Instruction::BitwiseNot(self.compile_unary(op, out)))
            }
            gpu::Bitwise::LeadingZeros(op) => {
                instructions.push(Instruction::LeadingZeros(self.compile_unary(op, out)))
            }
            gpu::Bitwise::FindFirstSet(op) => {
                let instruction = Instruction::FindFirstSet(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
        };
    }

    fn compile_operator(
        &mut self,
        value: gpu::Operator,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Operator::Index(op) | gpu::Operator::UncheckedIndex(op) => {
                instructions.push(Instruction::Index(self.compile_index(op, out)));
            }
            gpu::Operator::IndexAssign(op) | gpu::Operator::UncheckedIndexAssign(op) => {
                instructions.push(Instruction::IndexAssign(self.compile_index_assign(op, out)));
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
                len: op.len.as_const().unwrap().as_u32(),
            }),
            gpu::Operator::Select(op) => instructions.push(Instruction::Select {
                cond: self.compile_variable(op.cond),
                then: self.compile_variable(op.then),
                or_else: self.compile_variable(op.or_else),
                out: self.compile_variable(out),
            }),
            // Needs special conversion semantics
            gpu::Operator::Cast(op)
                if is_fp4_fp6_fp8(op.input.elem()) || is_fp4_fp6_fp8(out.elem()) =>
            {
                let inst = self.compile_unary(op, out);

                // We may need these for intermediates
                self.flags.elem_f16 = true;
                self.flags.elem_bf16 = true;
                let vec = NonZero::new(inst.input.item().vectorization as u8);
                self.compile_item(gpu::Item::vectorized(gpu::Elem::Float(FloatKind::F16), vec));
                self.compile_item(gpu::Item::vectorized(
                    gpu::Elem::Float(FloatKind::BF16),
                    vec,
                ));

                instructions.push(Instruction::SpecialCast(inst));
            }
            gpu::Operator::Cast(op) => {
                instructions.push(Instruction::Assign(self.compile_unary(op, out)))
            }
            gpu::Operator::Reinterpret(op) => {
                instructions.push(Instruction::Bitcast(self.compile_unary(op, out)))
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

    fn compile_index_assign(
        &mut self,
        value: gpu::IndexAssignOperator,
        out: gpu::Variable,
    ) -> IndexAssignInstruction<D> {
        IndexAssignInstruction {
            index: self.compile_variable(value.index),
            value: self.compile_variable(value.value),
            line_size: value.line_size,
            out: self.compile_variable(out),
        }
    }

    fn compile_index(
        &mut self,
        value: gpu::IndexOperator,
        out: gpu::Variable,
    ) -> IndexInstruction<D> {
        IndexInstruction {
            list: self.compile_variable(value.list),
            index: self.compile_variable(value.index),
            line_size: value.line_size,
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
            gpu::VariableKind::GlobalScalar(id) => Variable::GlobalScalar {
                id,
                elem: self.compile_elem(item.elem),
                in_struct: self.compilation_options.grid_constants,
            },
            gpu::VariableKind::TensorMap(id) => {
                self.flags.inst_tma = true;
                Variable::TensorMap(id)
            }
            gpu::VariableKind::LocalMut { id } => Variable::LocalMut {
                id,
                item: self.compile_item(item),
            },
            gpu::VariableKind::Versioned { id, .. } => Variable::LocalMut {
                id,
                item: self.compile_item(item),
            },
            gpu::VariableKind::LocalConst { id } => Variable::LocalConst {
                id,
                item: self.compile_item(item),
            },
            gpu::VariableKind::GlobalOutputArray(id) => {
                Variable::GlobalOutputArray(id, self.compile_item(item))
            }
            gpu::VariableKind::ConstantScalar(value) => {
                Variable::ConstantScalar(value, self.compile_elem(value.elem()))
            }
            gpu::VariableKind::SharedMemory {
                id,
                length,
                alignment,
            } => {
                let item = self.compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == id) {
                    self.shared_memories
                        .push(SharedMemory::new(id, item, length, alignment));
                }
                Variable::SharedMemory(id, item, length)
            }
            gpu::VariableKind::ConstantArray { id, length } => {
                let item = self.compile_item(item);
                Variable::ConstantArray(id, item, length)
            }
            gpu::VariableKind::Builtin(builtin) => match builtin {
                gpu::Builtin::AbsolutePos => {
                    self.flags.indexes.absolute_pos = true;
                    Variable::AbsolutePos
                }
                gpu::Builtin::CubePosCluster if self.compilation_options.supports_clusters => {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterRank
                }
                gpu::Builtin::CubePosClusterX if self.compilation_options.supports_clusters => {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterIndexX
                }
                gpu::Builtin::CubePosClusterY if self.compilation_options.supports_clusters => {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterIndexY
                }
                gpu::Builtin::CubePosClusterZ if self.compilation_options.supports_clusters => {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterIndexZ
                }
                // Fallback if clusters aren't supported, ID is always 0 since clusters are always
                // (1, 1, 1) if unsupported
                gpu::Builtin::CubePosCluster
                | gpu::Builtin::CubePosClusterX
                | gpu::Builtin::CubePosClusterY
                | gpu::Builtin::CubePosClusterZ => const_u32(0),
                gpu::Builtin::AbsolutePosX => {
                    self.flags.indexes.absolute_pos_tuple = true;
                    Variable::AbsolutePosX
                }
                gpu::Builtin::AbsolutePosY => {
                    self.flags.indexes.absolute_pos_tuple = true;
                    Variable::AbsolutePosY
                }
                gpu::Builtin::AbsolutePosZ => {
                    self.flags.indexes.absolute_pos_tuple = true;
                    Variable::AbsolutePosZ
                }
                gpu::Builtin::CubeDim => {
                    self.flags.indexes.cube_dim = true;
                    Variable::CubeDim
                }
                gpu::Builtin::CubeDimX => {
                    self.flags.indexes.cube_dim_tuple = true;
                    Variable::CubeDimX
                }
                gpu::Builtin::CubeDimY => {
                    self.flags.indexes.cube_dim_tuple = true;
                    Variable::CubeDimY
                }
                gpu::Builtin::CubeDimZ => {
                    self.flags.indexes.cube_dim_tuple = true;
                    Variable::CubeDimZ
                }
                gpu::Builtin::CubeClusterDim => const_u32(self.cluster_dim.num_elems()),
                gpu::Builtin::CubeClusterDimX => const_u32(self.cluster_dim.x),
                gpu::Builtin::CubeClusterDimY => const_u32(self.cluster_dim.y),
                gpu::Builtin::CubeClusterDimZ => const_u32(self.cluster_dim.z),
                gpu::Builtin::CubePos => {
                    self.flags.indexes.cube_pos = true;
                    Variable::CubePos
                }
                gpu::Builtin::CubePosX => {
                    self.flags.indexes.cube_pos_tuple = true;
                    Variable::CubePosX
                }
                gpu::Builtin::CubePosY => {
                    self.flags.indexes.cube_pos_tuple = true;
                    Variable::CubePosY
                }
                gpu::Builtin::CubePosZ => {
                    self.flags.indexes.cube_pos_tuple = true;
                    Variable::CubePosZ
                }
                gpu::Builtin::CubeCount => {
                    self.flags.indexes.cube_count = true;
                    Variable::CubeCount
                }
                gpu::Builtin::CubeCountX => {
                    self.flags.indexes.cube_count_tuple = true;
                    Variable::CubeCountX
                }
                gpu::Builtin::CubeCountY => {
                    self.flags.indexes.cube_count_tuple = true;
                    Variable::CubeCountY
                }
                gpu::Builtin::CubeCountZ => {
                    self.flags.indexes.cube_count_tuple = true;
                    Variable::CubeCountZ
                }
                gpu::Builtin::UnitPos => {
                    self.flags.indexes.unit_pos = true;
                    Variable::UnitPos
                }
                gpu::Builtin::UnitPosX => {
                    self.flags.indexes.unit_pos_tuple = true;
                    Variable::UnitPosX
                }
                gpu::Builtin::UnitPosY => {
                    self.flags.indexes.unit_pos_tuple = true;
                    Variable::UnitPosY
                }
                gpu::Builtin::UnitPosZ => {
                    self.flags.indexes.unit_pos_tuple = true;
                    Variable::UnitPosZ
                }
                gpu::Builtin::PlaneDim => {
                    self.flags.indexes.plane_dim = true;
                    Variable::PlaneDim
                }
                gpu::Builtin::UnitPosPlane => {
                    self.flags.indexes.unit_pos_plane = true;
                    Variable::UnitPosPlane
                }
            },
            gpu::VariableKind::LocalArray { id, length } => {
                let item = self.compile_item(item);
                if !self.local_arrays.iter().any(|s| s.index == id) {
                    self.local_arrays.push(LocalArray::new(id, item, length));
                }
                Variable::LocalArray(id, item, length)
            }
            gpu::VariableKind::Matrix { id, mat } => {
                self.flags.inst_wmma = true;
                Variable::WmmaFragment {
                    id,
                    frag: self.compile_matrix(mat),
                }
            }
            gpu::VariableKind::Pipeline {
                id,
                item,
                num_stages,
            } => {
                self.flags.op_pipeline = true;
                let pipeline = Variable::Pipeline {
                    id,
                    item: self.compile_item(item),
                };
                if !self.pipelines.iter().any(|s| s.pipeline_id() == id) {
                    self.pipelines.push(PipelineOps::Init {
                        pipeline,
                        num_stages,
                    });
                }
                pipeline
            }
            gpu::VariableKind::Barrier { id, item, level } => {
                self.flags.op_barrier = true;
                match level {
                    gpu::BarrierLevel::CubeCoop(_) | gpu::BarrierLevel::CubeManual(_) => {
                        self.flags.indexes.cube_dim = true;
                        self.flags.indexes.unit_pos = true;
                    }
                    _ => {}
                }
                Variable::Barrier {
                    id,
                    item: self.compile_item(item),
                    level,
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

    fn compile_binding(&mut self, binding: cubecl_core::compute::Binding) -> Binding<D> {
        Binding {
            id: binding.id,
            item: self.compile_item(binding.item),
            location: binding.location,
            size: binding.size,
            vis: binding.visibility,
        }
    }

    fn compile_item(&mut self, item: gpu::Item) -> Item<D> {
        let item = Item::new(
            self.compile_elem(item.elem),
            item.vectorization.map(NonZero::get).unwrap_or(1).into(),
            false,
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
                gpu::FloatKind::E2M1 => {
                    self.flags.elem_fp4 = true;
                    Elem::FP4(FP4Kind::E2M1)
                }
                gpu::FloatKind::E2M3 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E2M3)
                }
                gpu::FloatKind::E3M2 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E3M2)
                }
                gpu::FloatKind::E4M3 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::E4M3)
                }
                gpu::FloatKind::E5M2 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::E5M2)
                }
                gpu::FloatKind::UE8M0 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::UE8M0)
                }
                gpu::FloatKind::F16 => {
                    self.flags.elem_f16 = true;
                    Elem::F16
                }
                gpu::FloatKind::BF16 => {
                    self.flags.elem_bf16 = true;
                    Elem::BF16
                }
                gpu::FloatKind::TF32 => Elem::TF32,
                gpu::FloatKind::Flex32 => Elem::F32,
                gpu::FloatKind::F32 => Elem::F32,
                gpu::FloatKind::F64 => Elem::F64,
            },
            gpu::Elem::AtomicFloat(kind) => match kind {
                gpu::FloatKind::F16 => Elem::Atomic(AtomicKind::F16),
                gpu::FloatKind::BF16 => Elem::Atomic(AtomicKind::BF16),
                gpu::FloatKind::F32 => Elem::Atomic(AtomicKind::F32),
                gpu::FloatKind::F64 => Elem::Atomic(AtomicKind::F64),
                kind => unimplemented!("atomic<{kind:?}> not yet supported"),
            },
            gpu::Elem::Int(kind) => match kind {
                gpu::IntKind::I8 => Elem::I8,
                gpu::IntKind::I16 => Elem::I16,
                gpu::IntKind::I32 => Elem::I32,
                gpu::IntKind::I64 => Elem::I64,
            },
            gpu::Elem::AtomicInt(kind) => match kind {
                gpu::IntKind::I32 => Elem::Atomic(AtomicKind::I32),
                gpu::IntKind::I64 => Elem::Atomic(AtomicKind::I64),
                kind => panic!("atomic<{kind:?}> isn't supported yet"),
            },
            gpu::Elem::UInt(kind) => match kind {
                gpu::UIntKind::U8 => Elem::U8,
                gpu::UIntKind::U16 => Elem::U16,
                gpu::UIntKind::U32 => Elem::U32,
                gpu::UIntKind::U64 => Elem::U64,
            },
            gpu::Elem::AtomicUInt(kind) => match kind {
                gpu::UIntKind::U32 => Elem::Atomic(AtomicKind::U32),
                gpu::UIntKind::U64 => Elem::Atomic(AtomicKind::U64),
                kind => unimplemented!("atomic<{kind:?}> not yet supported"),
            },
            gpu::Elem::Bool => Elem::Bool,
        }
    }
}

fn is_fp4_fp6_fp8(elem: gpu::Elem) -> bool {
    match elem {
        gpu::Elem::Float(kind) => matches!(
            kind,
            FloatKind::E2M1
                | FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0
        ),
        _ => false,
    }
}

fn const_u32<D: Dialect>(value: u32) -> Variable<D> {
    Variable::ConstantScalar(
        gpu::ConstantScalarValue::UInt(value as u64, UIntKind::U32),
        Elem::U32,
    )
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
        gpu::Elem::AtomicInt(gpu::IntKind::I64),
        gpu::Elem::AtomicUInt(gpu::UIntKind::U32),
        gpu::Elem::AtomicUInt(gpu::UIntKind::U64),
        gpu::Elem::Float(gpu::FloatKind::BF16),
        gpu::Elem::Float(gpu::FloatKind::F16),
        gpu::Elem::Float(gpu::FloatKind::F32),
        gpu::Elem::Float(gpu::FloatKind::Flex32),
        gpu::Elem::AtomicFloat(gpu::FloatKind::F32),
        // Causes CUDA_ERROR_INVALID_VALUE for matmul, disabling until that can be investigated
        //gpu::Elem::Float(gpu::FloatKind::F64),
        gpu::Elem::Bool,
    ];

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}
