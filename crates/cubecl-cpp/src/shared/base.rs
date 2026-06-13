use super::{
    BinaryInstruction, Body, Component, ComputeKernel, Dialect, Elem, FP4Kind, FP6Kind, FP8Kind,
    FragmentIdent, FragmentLayout, FragmentType, IndexInstruction, Instruction, Item, KernelArg,
    SharedMemory, UnaryInstruction, Value, WarpInstruction, WmmaInstruction, barrier::BarrierOps,
};
use crate::shared::{Builtin, MmaShape, PointerClass};
use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    CubeDim,
    ir::{
        self as ir, AddressSpace, DeviceProperties, ElemType, FloatKind, InstructionModes,
        OpaqueType, Operation, Processor, SourceLoc, StorageType, Type,
        features::{AtomicUsage, EnumSet, TypeUsage},
    },
    post_processing::{self, checked_io::CheckedIoVisitor, disaggregate::DisaggregateVisitor},
    prelude::{FastMath, KernelDefinition, Visibility},
    server::ExecutionMode,
};
use cubecl_opt::{Optimizer, SharedLiveness};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

pub(super) static COUNTER_TMP_VAR: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

#[derive(Clone, Debug)]
pub struct CompilationOptions {
    pub warp_size: u32,
    pub supports_features: CppSupportedFeatures,
}

#[derive(Clone, Debug, Default)]
pub struct CppSupportedFeatures {
    pub grid_constants: bool,
    pub clusters: bool,
    pub fast_math: bool,
    pub fast_tanh: bool,
    pub elect_sync: bool,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            warp_size: 32,
            supports_features: Default::default(),
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
    pub plane_pos: bool,
    pub unit_pos: bool,
    pub unit_pos_tuple: bool,
    pub unit_pos_plane: bool,
    pub cluster_pos: bool,
}

/// Flags gathered during Cube IR translation for the kernel compilation.
#[derive(Debug, Clone)]
pub struct Flags<D: Dialect> {
    pub elem_fp4: bool,
    pub elem_fp6: bool,
    pub elem_fp8: bool,
    pub elem_bf16: bool,
    pub elem_f16: bool,
    pub elem_tf32: bool,
    pub indexes: CubeIndexFlags,
    pub op_barrier: bool,
    pub thread_block: bool,
    pub inst_tma: bool,
    pub inst_tma_im2col: bool,
    pub inst_wmma: bool,
    pub inst_ptx_wrappers: bool,
    pub inst_async_copy: bool,
    pub use_grid_constants: bool,
    pub static_meta_length: usize,
    pub has_dynamic_meta: bool,
    pub has_info: bool,
    pub cube_dim: CubeDim,
    pub cluster_dim: Option<CubeDim>,
    pub address_type: Item<D>,
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug)]
pub struct CppCompiler<D: Dialect> {
    kernel_name: String,
    buffer_vis: Vec<Visibility>,
    barriers: Vec<BarrierOps<D>>,
    compilation_options: CompilationOptions,
    ext_meta_positions: HashMap<ir::Value, u32>,
    cluster_dim: CubeDim,
    extensions: Vec<D::Extension>,
    flags: Flags<D>,
    items: HashSet<Item<D>>,
    info: cubecl_core::Info,
    source_loc: Option<SourceLoc>,
    strategy: ExecutionMode,
    addr_type: Item<D>,
}

impl<D: Dialect> Default for Flags<D> {
    fn default() -> Self {
        Self {
            elem_fp4: Default::default(),
            elem_fp6: Default::default(),
            elem_fp8: Default::default(),
            elem_bf16: Default::default(),
            elem_f16: Default::default(),
            elem_tf32: Default::default(),
            indexes: Default::default(),
            op_barrier: Default::default(),
            thread_block: Default::default(),
            inst_tma: Default::default(),
            inst_tma_im2col: Default::default(),
            inst_wmma: Default::default(),
            inst_ptx_wrappers: Default::default(),
            inst_async_copy: Default::default(),
            use_grid_constants: Default::default(),
            static_meta_length: Default::default(),
            has_info: Default::default(),
            has_dynamic_meta: Default::default(),
            cube_dim: CubeDim::new_single(),
            cluster_dim: Default::default(),
            address_type: Item::Scalar(Elem::U32),
        }
    }
}

impl<D: Dialect> Default for CppCompiler<D> {
    fn default() -> Self {
        Self {
            kernel_name: Default::default(),
            buffer_vis: Default::default(),
            barriers: Default::default(),
            compilation_options: Default::default(),
            ext_meta_positions: Default::default(),
            cluster_dim: CubeDim::new_single(),
            extensions: Default::default(),
            flags: Flags::default(),
            items: Default::default(),
            info: Default::default(),
            source_loc: Default::default(),
            strategy: Default::default(),
            addr_type: Item::Scalar(Elem::U32),
        }
    }
}

impl<D: Dialect> Compiler for CppCompiler<D> {
    type Representation = ComputeKernel<D>;
    type CompilationOptions = CompilationOptions;

    fn compile(
        &mut self,
        mut kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        strategy: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile cpp kernel\nCaused by:\n  ".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        self.addr_type = self.compile_type(addr_type.into());
        self.compilation_options = compilation_options.clone();
        self.strategy = strategy;
        self.kernel_name = kernel.options.kernel_name.clone();

        if !self.compilation_options.supports_features.clusters {
            kernel.options.cluster_dim = None;
        }
        self.cluster_dim = kernel.options.cluster_dim.unwrap_or(CubeDim::new_single());

        let ir = self.clone().compile_ir(kernel, addr_type);
        COUNTER_TMP_VAR.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(ir)
    }

    fn elem_size(&self, elem: ir::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "cpp"
    }
}

impl<D: Dialect> CppCompiler<D> {
    fn compile_ir(
        mut self,
        value: KernelDefinition,
        address_type: StorageType,
    ) -> ComputeKernel<D> {
        let metadata = self.build_metadata(&value);
        self.info = cubecl_core::Info::new(&value.scalars, metadata, address_type);

        let scope_state = value.body.state().clone_deep();

        let mut opt = Optimizer::shared_only(value.body.clone(), value.cube_dim);
        let shared_allocs = opt.main.analysis::<SharedLiveness>(&opt.global_state);

        *value.body.state_mut() = scope_state;

        CheckedIoVisitor::new(self.strategy, self.kernel_name.clone()).apply(&value.body);
        DisaggregateVisitor::apply(&value.body);

        self.buffer_vis = post_processing::optimize_scope(&value.body).into();
        self.buffer_vis
            .resize(value.num_global_buffers(), Visibility::Read);

        let address_type = self.compile_type(address_type.into());
        let instructions = self.compile_scope(&value.body);

        let tensor_maps = value
            .tensor_maps
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let buffers = value
            .buffers
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let scalars = value
            .scalars
            .into_iter()
            .map(|binding| (self.compile_storage_type(binding.ty), binding.count))
            .collect::<Vec<_>>();

        let shared_memories = shared_allocs
            .allocations
            .values()
            .map(|alloc| SharedMemory {
                ptr: self.compile_value(ir::Value::new(alloc.id, alloc.smem.root_ptr.ty)),
                value_ty: self.compile_type(alloc.smem.value_ty),
                align: alloc.smem.alignment,
                offset: alloc.offset,
            })
            .collect();

        let body = Body {
            instructions,
            shared_memories,
            barriers: self.barriers,
            info_by_ptr: !self.compilation_options.supports_features.grid_constants,
            has_dynamic_meta: self.info.has_dynamic_meta,
            address_type: self.addr_type,
        };

        // translation flags
        let flags = Flags {
            indexes: D::builtin_rules(&self.flags.indexes),
            inst_wmma: self.flags.inst_wmma,
            thread_block: self.flags.thread_block,
            op_barrier: self.flags.op_barrier,
            elem_fp4: self.flags.elem_fp4,
            elem_fp6: self.flags.elem_fp6,
            elem_fp8: self.flags.elem_fp8,
            elem_bf16: self.flags.elem_bf16,
            elem_f16: self.flags.elem_f16,
            elem_tf32: self.flags.elem_tf32,
            inst_tma: self.flags.inst_tma,
            inst_tma_im2col: self.flags.inst_tma_im2col,
            inst_async_copy: self.flags.inst_async_copy,
            inst_ptx_wrappers: self.flags.inst_ptx_wrappers,
            use_grid_constants: self.compilation_options.supports_features.grid_constants,
            has_info: self.info.has_info(),
            has_dynamic_meta: self.info.has_dynamic_meta,
            static_meta_length: self.info.metadata.static_len() as usize,
            cube_dim: value.cube_dim,
            cluster_dim: value.options.cluster_dim,
            address_type,
        };

        let mut cluster_dim = value.options.cluster_dim;
        if !self.compilation_options.supports_features.clusters {
            cluster_dim = None;
        }

        ComputeKernel {
            tensor_maps,
            buffers,
            scalars,
            meta_static_len: self.info.metadata.static_len() as usize,
            cube_dim: value.cube_dim,
            body,
            extensions: self.extensions,
            flags,
            items: self.items,
            kernel_name: value.options.kernel_name,
            cluster_dim,
            info: self.info.clone(),
        }
    }

    fn build_metadata(&mut self, value: &KernelDefinition) -> cubecl_core::Metadata {
        let mut num_ext = 0;

        let mut all_meta: Vec<_> = value
            .buffers
            .iter()
            .chain(value.tensor_maps.iter())
            .map(|buf| (buf.id, buf.value, buf.has_extended_meta))
            .collect();

        all_meta.sort_by_key(|(id, _, _)| *id);

        for (_, value, has_extended_meta) in &all_meta {
            self.ext_meta_positions.insert(*value, num_ext);
            if *has_extended_meta {
                num_ext += 1;
            }
        }

        let num_meta = all_meta.len();

        cubecl_core::Metadata::new(num_meta as u32, num_ext)
    }

    pub(crate) fn ext_meta_position(&self, val: &ir::Value) -> u32 {
        self.ext_meta_positions[val]
    }

    fn compile_scope(&mut self, scope: &ir::Scope) -> Vec<Instruction<D>> {
        let mut instructions = Vec::new();

        let dialect_processors = D::processors();
        let mut processors: Vec<&dyn Processor> = vec![];
        processors.extend(dialect_processors.iter().map(|it| &**it));

        let processing = scope.process(processors);

        processing
            .instructions
            .into_iter()
            .for_each(|op| self.compile_instruction(&mut instructions, op));

        instructions
    }

    fn compile_instruction(
        &mut self,
        instructions: &mut Vec<Instruction<D>>,
        instruction: ir::Instruction,
    ) {
        self.update_debug_loc(instructions, &instruction);
        let out = instruction.out;

        match instruction.operation {
            ir::Operation::Copy(value) => {
                instructions.push(Instruction::Assign(UnaryInstruction {
                    input: self.compile_value(value),
                    out: self.compile_value(out.unwrap()),
                }));
            }
            ir::Operation::DeclareVariable {
                addr_space: AddressSpace::Local,
                value_ty,
                ..
            } => instructions.push(Instruction::DeclareVariable {
                val: self.compile_value(out.unwrap()),
                value_ty: self.compile_type(value_ty),
            }),
            ir::Operation::DeclareVariable {
                addr_space: AddressSpace::Shared,
                ..
            } => {
                // Optimizer handles parsing and allocating these
            }
            ir::Operation::DeclareVariable { addr_space, .. } => {
                unimplemented!("Unsupported declaration address space {addr_space}")
            }
            ir::Operation::Arithmetic(op) => {
                self.compile_arithmetic(op, out, instruction.modes, instructions)
            }
            ir::Operation::Memory(op) => self.compile_memory(op, out, instructions),
            ir::Operation::Comparison(op) => self.compile_comparison(op, out, instructions),
            ir::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
            ir::Operation::Operator(op) => self.compile_operator(op, out, instructions),
            ir::Operation::Atomic(op) => self.compile_atomic(op, out, instructions),
            ir::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            ir::Operation::Branch(val) => self.compile_branch(instructions, val),
            ir::Operation::Synchronization(val) => match val {
                ir::Synchronization::SyncCube => instructions.push(Instruction::SyncThreads),
                ir::Synchronization::SyncPlane => instructions.push(Instruction::SyncWarp),
                ir::Synchronization::SyncStorage => instructions.push(Instruction::SyncThreads),
                ir::Synchronization::SyncAsyncProxyShared => {
                    self.flags.inst_tma = true;
                    instructions.push(Instruction::ProxyAsyncToSharedFence)
                }
            },
            ir::Operation::WorkgroupUniformLoad(input) => {
                let is_atomic = input.ty.is_atomic();
                instructions.push(Instruction::SyncThreads);
                let load = UnaryInstruction {
                    input: self.compile_value(input),
                    out: self.compile_value(out.unwrap()),
                };
                if is_atomic {
                    instructions.push(Instruction::AtomicLoad(load));
                } else {
                    instructions.push(Instruction::Load(load));
                }
            }
            ir::Operation::Plane(op) => {
                self.flags.indexes.plane_dim_checked = true;
                let out = self.compile_value(out.unwrap());
                match op {
                    ir::Plane::Sum(op) => {
                        let instruction = WarpInstruction::ReduceSum {
                            input: self.compile_value(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction));
                    }
                    ir::Plane::InclusiveSum(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::InclusiveSum {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::InclusiveProd(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::InclusiveProd {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::ExclusiveSum(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::ExclusiveSum {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::ExclusiveProd(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::ExclusiveProd {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::Prod(op) => {
                        let instruction = WarpInstruction::ReduceProd {
                            input: self.compile_value(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    ir::Plane::Max(op) => {
                        let instruction = WarpInstruction::ReduceMax {
                            input: self.compile_value(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    ir::Plane::Min(op) => {
                        let instruction = WarpInstruction::ReduceMin {
                            input: self.compile_value(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    ir::Plane::Elect => {
                        if self.compilation_options.supports_features.elect_sync {
                            self.flags.inst_ptx_wrappers = true;
                            instructions.push(Instruction::Warp(WarpInstruction::Elect { out }))
                        } else {
                            instructions
                                .push(Instruction::Warp(WarpInstruction::ElectFallback { out }))
                        }
                    }
                    ir::Plane::All(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::All {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::Any(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Any {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::Ballot(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Ballot {
                            input: self.compile_value(op.input),
                            out,
                        }))
                    }
                    ir::Plane::Broadcast(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Broadcast {
                            input: self.compile_value(op.lhs),
                            id: self.compile_value(op.rhs),
                            out,
                        }))
                    }
                    ir::Plane::Shuffle(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Shuffle {
                            input: self.compile_value(op.lhs),
                            src_lane: self.compile_value(op.rhs),
                            out,
                        }))
                    }
                    ir::Plane::ShuffleXor(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::ShuffleXor {
                            input: self.compile_value(op.lhs),
                            mask: self.compile_value(op.rhs),
                            out,
                        }))
                    }
                    ir::Plane::ShuffleUp(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::ShuffleUp {
                            input: self.compile_value(op.lhs),
                            delta: self.compile_value(op.rhs),
                            out,
                        }))
                    }
                    ir::Plane::ShuffleDown(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::ShuffleDown {
                            input: self.compile_value(op.lhs),
                            delta: self.compile_value(op.rhs),
                            out,
                        }))
                    }
                }
            }
            ir::Operation::CoopMma(cmma) => instructions.push(self.compile_cmma(cmma, out)),
            ir::Operation::NonSemantic(debug) => match debug {
                ir::NonSemantic::Print {
                    format_string,
                    args,
                } => instructions.push(Instruction::Printf {
                    format_string,
                    args: args
                        .into_iter()
                        .map(|arg| self.compile_value(arg))
                        .collect(),
                }),
                ir::NonSemantic::Comment { content } => {
                    instructions.push(Instruction::Comment { content })
                }
                // Don't need to handle scopes
                _ => {}
            },
            ir::Operation::TensorIndexing(_) => panic!("Tensor indexing only supported in Vulkan"),
            ir::Operation::Barrier(barrier_ops) => match barrier_ops {
                ir::BarrierOps::Init {
                    barrier,
                    is_elected,
                    arrival_count,
                } => {
                    let Type::Opaque(OpaqueType::Barrier(level)) = barrier.ty else {
                        unreachable!()
                    };
                    let barrier = self.compile_value(barrier);
                    let arrival_count = self.compile_value(arrival_count);
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Init {
                        barrier,
                        is_elected: self.compile_value(is_elected),
                        arrival_count,
                        level,
                    }));
                }
                ir::BarrierOps::InitManual {
                    barrier,
                    arrival_count,
                } => {
                    let barrier = self.compile_value(barrier);
                    let arrival_count = self.compile_value(arrival_count);
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::InitManual {
                            barrier,
                            arrival_count,
                        },
                    ));
                }
                ir::BarrierOps::MemCopyAsync {
                    barrier,
                    source,
                    destination,
                    source_length,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsync {
                            barrier: self.compile_value(barrier),
                            source: self.compile_value(source),
                            destination: self.compile_value(destination),
                            source_length: self.compile_value(source_length),
                            cooperative: false,
                        },
                    ));
                }
                ir::BarrierOps::MemCopyAsyncCooperative {
                    barrier,
                    source,
                    destination,
                    source_length,
                } => {
                    self.flags.thread_block = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsync {
                            barrier: self.compile_value(barrier),
                            source: self.compile_value(source),
                            destination: self.compile_value(destination),
                            source_length: self.compile_value(source_length),
                            cooperative: true,
                        },
                    ));
                }
                ir::BarrierOps::MemCopyAsyncTx {
                    barrier,
                    source,
                    destination,
                    source_length,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsyncTx {
                            barrier: self.compile_value(barrier),
                            source: self.compile_value(source),
                            destination: self.compile_value(destination),
                            source_length: self.compile_value(source_length),
                        },
                    ));
                }
                ir::BarrierOps::CopyAsync {
                    source,
                    destination,
                    source_length,
                    copy_length,
                    checked,
                } => {
                    self.flags.inst_async_copy = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::CopyAsync {
                            source: self.compile_value(source),
                            destination: self.compile_value(destination),
                            source_length: self.compile_value(source_length),
                            copy_size: copy_length,
                            checked,
                        },
                    ));
                }
                ir::BarrierOps::TmaLoad {
                    barrier,
                    tensor_map,
                    destination,
                    indices,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsyncTensorGlobalToShared {
                            barrier: self.compile_value(barrier),
                            smem_buffer: self.compile_value(destination),
                            tensor_map: self.compile_value(tensor_map),
                            indices: indices
                                .into_iter()
                                .map(|it| self.compile_value(it))
                                .collect(),
                        },
                    ));
                }
                ir::BarrierOps::TmaLoadIm2col {
                    barrier,
                    tensor_map,
                    destination,
                    indices,
                    offsets,
                } => {
                    self.flags.inst_tma_im2col = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::TmaLoadIm2col {
                            barrier: self.compile_value(barrier),
                            smem_buffer: self.compile_value(destination),
                            tensor_map: self.compile_value(tensor_map),
                            indices: indices
                                .into_iter()
                                .map(|it| self.compile_value(it))
                                .collect(),
                            offsets: offsets
                                .into_iter()
                                .map(|it| self.compile_value(it))
                                .collect(),
                        },
                    ));
                }
                ir::BarrierOps::Arrive { barrier } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Arrive {
                        barrier: self.compile_value(barrier),
                        token: self.compile_value(out.unwrap()),
                    }))
                }
                ir::BarrierOps::ArriveTx {
                    barrier,
                    arrive_count_update,
                    transaction_count_update,
                } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::ArriveTx {
                        barrier: self.compile_value(barrier),
                        token: self.compile_value(out.unwrap()),
                        arrive_count_update: self.compile_value(arrive_count_update),
                        transaction_count_update: self.compile_value(transaction_count_update),
                    }))
                }
                ir::BarrierOps::CommitCopyAsync { barrier } => {
                    self.flags.inst_async_copy = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::ArriveCopyAsync {
                            barrier: self.compile_value(barrier),
                        },
                    ))
                }
                ir::BarrierOps::ExpectTx {
                    barrier,
                    transaction_count_update,
                } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::ExpectTx {
                        barrier: self.compile_value(barrier),
                        transaction_count_update: self.compile_value(transaction_count_update),
                    }))
                }
                ir::BarrierOps::Wait { barrier, token } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Wait {
                        barrier: self.compile_value(barrier),
                        token: self.compile_value(token),
                    }))
                }
                ir::BarrierOps::WaitParity { barrier, phase } => instructions.push(
                    Instruction::Barrier(super::barrier::BarrierOps::WaitParity {
                        barrier: self.compile_value(barrier),
                        phase: self.compile_value(phase),
                    }),
                ),
                ir::BarrierOps::ArriveAndWait { barrier } => {
                    let Type::Opaque(OpaqueType::Barrier(level)) = barrier.ty else {
                        unreachable!()
                    };
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::ArriveAndWait {
                            barrier: self.compile_value(barrier),
                            level,
                        },
                    ))
                }
            },
            ir::Operation::Tma(tma_ops) => {
                self.flags.inst_tma = true;
                match tma_ops {
                    ir::TmaOps::TmaStore {
                        source,
                        coordinates,
                    } => {
                        instructions.push(Instruction::MemCopyAsyncTensorSharedToGlobal {
                            smem_buffer: self.compile_value(source),
                            tensor_map: self.compile_value(out.unwrap()),
                            indices: coordinates
                                .into_iter()
                                .map(|it| self.compile_value(it))
                                .collect(),
                        });
                    }
                    ir::TmaOps::CommitGroup => {
                        instructions.push(Instruction::BulkCommitGroup);
                    }
                    ir::TmaOps::WaitGroup { max_pending } => {
                        instructions.push(Instruction::BulkWaitGroup { max_pending });
                    }
                    ir::TmaOps::WaitGroupRead { max_pending } => {
                        instructions.push(Instruction::BulkWaitGroupRead { max_pending });
                    }
                }
            }
            ir::Operation::Marker(_) => {}
            ir::Operation::ConstructAggregate(..) | ir::Operation::ExtractAggregateField(..) => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    fn update_debug_loc(&mut self, instructions: &mut Vec<Instruction<D>>, inst: &ir::Instruction) {
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

    fn compile_cmma(&mut self, cmma: ir::CoopMma, out: Option<ir::Value>) -> Instruction<D> {
        self.flags.inst_wmma = true;

        let inst = match cmma {
            ir::CoopMma::Fill { value } => WmmaInstruction::Fill {
                frag: self.compile_value(out.unwrap()),
                value: self.compile_value(value),
            },
            ir::CoopMma::Load {
                ptr,
                stride,
                layout,
            } => WmmaInstruction::Load {
                frag: self.compile_value(out.unwrap()),
                ptr: self.compile_value(ptr),
                stride: self.compile_value(stride),
                layout: layout.and_then(|l| self.compile_matrix_layout(l)),
            },
            ir::CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => WmmaInstruction::Execute {
                frag_a: self.compile_value(mat_a),
                frag_b: self.compile_value(mat_b),
                frag_c: self.compile_value(mat_c),
                frag_d: self.compile_value(out.unwrap()),
                warp_size: self.compilation_options.warp_size,
            },
            ir::CoopMma::ExecuteManual {
                matrix,
                registers_a,
                registers_b,
                registers_c,
            } => WmmaInstruction::ExecuteManual {
                shape: MmaShape::new(matrix.m as u32, matrix.n as u32, matrix.k as u32),
                frag_a: self.compile_value(registers_a),
                frag_b: self.compile_value(registers_b),
                frag_c: self.compile_value(registers_c),
                frag_d: self.compile_value(out.unwrap()),
            },
            ir::CoopMma::ExecuteScaled {
                matrix,
                registers_a,
                registers_b,
                registers_c,
                scales_a,
                scales_b,
                scales_factor,
            } => WmmaInstruction::ExecuteScaled {
                shape: MmaShape::new(matrix.m as u32, matrix.n as u32, matrix.k as u32),
                frag_a: self.compile_value(registers_a),
                frag_b: self.compile_value(registers_b),
                frag_c: self.compile_value(registers_c),
                frag_d: self.compile_value(out.unwrap()),

                scales_a: self.compile_value(scales_a),
                scales_b: self.compile_value(scales_b),
                scales_factor: scales_factor as u32,
            },
            ir::CoopMma::ExecuteElementwise { .. } => {
                panic!("Elementwise only supported in Vulkan")
            }
            ir::CoopMma::Store {
                mat,
                stride,
                destination,
                layout,
            } => {
                self.flags.indexes.unit_pos = true;
                self.flags.indexes.plane_pos = true;
                WmmaInstruction::Store {
                    destination: self.compile_value(destination),
                    frag: self.compile_value(mat),
                    stride: self.compile_value(stride),
                    layout: self
                        .compile_matrix_layout(layout)
                        .expect("Layout required for store instruction"),
                }
            }
            ir::CoopMma::LoadMatrix {
                ptr,
                factor,
                transpose,
            } => WmmaInstruction::LdMatrix {
                output: self.compile_value(out.unwrap()),
                ptr: self.compile_value(ptr),
                factor: factor as u32,
                transpose,
            },
            ir::CoopMma::StoreMatrix {
                registers,
                factor,
                transpose,
                destination,
            } => WmmaInstruction::StMatrix {
                registers: self.compile_value(registers),
                ptr: self.compile_value(destination),
                factor: factor as u32,
                transpose,
            },
            ir::CoopMma::Cast { input } => WmmaInstruction::Cast {
                input: self.compile_value(input),
                output: self.compile_value(out.unwrap()),
            },
            ir::CoopMma::RowIndex { .. } | ir::CoopMma::ColIndex { .. } => {
                panic!("Row/Col index should be handled by processors")
            }
            ir::CoopMma::LoadTensor { .. } | ir::CoopMma::StoreTensor { .. } => {
                panic!("Load/store tensor is only supported in Vulkan")
            }
        };

        D::register_wmma_instruction_extension(&mut self.extensions, &inst);

        Instruction::Wmma(inst)
    }

    fn compile_metadata(
        &mut self,
        metadata: ir::Metadata,
        out: Option<ir::Value>,
    ) -> Instruction<D> {
        let out = out.unwrap();
        match metadata {
            ir::Metadata::Stride { dim, list } => {
                let position = self.ext_meta_position(&list);
                let offset = self.info.metadata.stride_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_value(offset.into()),
                    dim: self.compile_value(dim),
                    out: self.compile_value(out),
                }
            }
            ir::Metadata::Shape { dim, list } => {
                let position = self.ext_meta_position(&list);
                let offset = self.info.metadata.shape_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_value(offset.into()),
                    dim: self.compile_value(dim),
                    out: self.compile_value(out),
                }
            }
            ir::Metadata::BufferLength { list } => {
                let out = self.compile_value(out);

                let AddressSpace::Global(id) = list.address_space() else {
                    unreachable!("Value should have id")
                };
                let offset = self.info.metadata.buffer_len_index(id);
                Instruction::Metadata {
                    info_offset: self.compile_value(offset.into()),
                    out,
                }
            }
        }
    }

    fn compile_branch(&mut self, instructions: &mut Vec<Instruction<D>>, branch: ir::Branch) {
        match branch {
            ir::Branch::If(op) => instructions.push(Instruction::If {
                cond: self.compile_value(op.cond),
                instructions: self.compile_scope(&op.scope),
            }),
            ir::Branch::IfElse(op) => instructions.push(Instruction::IfElse {
                cond: self.compile_value(op.cond),
                instructions_if: self.compile_scope(&op.scope_if),
                instructions_else: self.compile_scope(&op.scope_else),
            }),
            ir::Branch::Switch(op) => instructions.push(Instruction::Switch {
                value: self.compile_value(op.value),
                instructions_default: self.compile_scope(&op.scope_default),
                instructions_cases: op
                    .cases
                    .into_iter()
                    .map(|(val, block)| (self.compile_value(val), self.compile_scope(&block)))
                    .collect(),
            }),
            ir::Branch::Return => instructions.push(Instruction::Return),
            ir::Branch::Break => instructions.push(Instruction::Break),
            ir::Branch::Unreachable => instructions.push(Instruction::Unreachable),
            ir::Branch::RangeLoop(range_loop) => instructions.push(Instruction::RangeLoop {
                i: self.compile_value(range_loop.i),
                start: self.compile_value(range_loop.start),
                end: self.compile_value(range_loop.end),
                step: range_loop.step.map(|it| self.compile_value(it)),
                inclusive: range_loop.inclusive,
                instructions: self.compile_scope(&range_loop.scope),
            }),
            ir::Branch::Loop(op) => instructions.push(Instruction::Loop {
                instructions: self.compile_scope(&op.scope),
            }),
        };
    }

    fn compile_atomic(
        &mut self,
        value: ir::AtomicOp,
        out: Option<ir::Value>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        match value {
            ir::AtomicOp::Load(ptr) => {
                instructions.push(Instruction::AtomicLoad(UnaryInstruction {
                    input: self.compile_value(ptr),
                    out: self.compile_value(out.unwrap()),
                }))
            }
            ir::AtomicOp::Store(op) => {
                instructions.push(Instruction::AtomicStore(UnaryInstruction {
                    input: self.compile_value(op.value),
                    out: self.compile_value(op.ptr),
                }))
            }
            ir::AtomicOp::Swap(op) => instructions.push(Instruction::AtomicSwap(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::Add(op) => instructions.push(Instruction::AtomicAdd(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::Sub(op) => instructions.push(Instruction::AtomicSub(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::Max(op) => instructions.push(Instruction::AtomicMax(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::Min(op) => instructions.push(Instruction::AtomicMin(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::And(op) => instructions.push(Instruction::AtomicAnd(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::Or(op) => instructions.push(Instruction::AtomicOr(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::Xor(op) => instructions.push(Instruction::AtomicXor(
                self.compile_atomic_binary(op, out.unwrap()),
            )),
            ir::AtomicOp::CompareAndSwap(op) => instructions.push(Instruction::AtomicCAS {
                input: self.compile_value(op.ptr),
                cmp: self.compile_value(op.cmp),
                val: self.compile_value(op.val),
                out: self.compile_value(out.unwrap()),
            }),
        }
    }

    fn compile_arithmetic(
        &mut self,
        value: ir::Arithmetic,
        out: Option<ir::Value>,
        modes: InstructionModes,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            ir::Arithmetic::Add(op) => {
                instructions.push(Instruction::Add(self.compile_binary(op, out)))
            }
            ir::Arithmetic::SaturatingAdd(op) => {
                instructions.push(Instruction::SaturatingAdd(self.compile_binary(op, out)))
            }
            ir::Arithmetic::Mul(op) => {
                instructions.push(Instruction::Mul(self.compile_binary(op, out)))
            }
            ir::Arithmetic::Div(op) => {
                let op = self.compile_binary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::AllowReciprocal
                        | FastMath::ReducedPrecision
                        | FastMath::UnsignedZero
                        | FastMath::NotInf,
                    Instruction::Div(op),
                    Instruction::FastDiv(op),
                ))
            }
            ir::Arithmetic::Sub(op) => {
                instructions.push(Instruction::Sub(self.compile_binary(op, out)))
            }
            ir::Arithmetic::SaturatingSub(op) => {
                instructions.push(Instruction::SaturatingSub(self.compile_binary(op, out)))
            }
            ir::Arithmetic::MulHi(op) => {
                let instruction = Instruction::HiMul(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Abs(op) => {
                instructions.push(Instruction::Abs(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Exp(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Exp(op),
                    Instruction::FastExp(op),
                ));
            }
            ir::Arithmetic::Log(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Log(op),
                    Instruction::FastLog(op),
                ));
            }
            ir::Arithmetic::Log1p(op) => {
                instructions.push(Instruction::Log1p(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Expm1(op) => {
                instructions.push(Instruction::Expm1(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Cos(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Cos(op),
                    Instruction::FastCos(op),
                ));
            }
            ir::Arithmetic::Sin(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Sin(op),
                    Instruction::FastSin(op),
                ));
            }
            ir::Arithmetic::Tan(op) => {
                instructions.push(Instruction::Tan(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Tanh(op) => {
                let op = self.compile_unary(op, out);
                let instruction = Instruction::Tanh(op);
                D::register_instruction_extension(&mut self.extensions, &instruction);
                if self.compilation_options.supports_features.fast_tanh {
                    instructions.push(self.select_fast_float(
                        out.ty,
                        modes,
                        FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                        instruction,
                        Instruction::FastTanh(op),
                    ))
                } else {
                    instructions.push(instruction);
                }
            }
            ir::Arithmetic::Sinh(op) => {
                let instruction = Instruction::Sinh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Cosh(op) => {
                let instruction = Instruction::Cosh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcCos(op) => {
                let instruction = Instruction::ArcCos(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcSin(op) => {
                let instruction = Instruction::ArcSin(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcTan(op) => {
                let instruction = Instruction::ArcTan(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcSinh(op) => {
                let instruction = Instruction::ArcSinh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcCosh(op) => {
                let instruction = Instruction::ArcCosh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcTanh(op) => {
                let instruction = Instruction::ArcTanh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Degrees(op) => {
                let instruction = Instruction::Degrees(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Radians(op) => {
                let instruction = Instruction::Radians(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::ArcTan2(op) => {
                let instruction = Instruction::ArcTan2(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Powf(op) => {
                let op = self.compile_binary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Powf(op),
                    Instruction::FastPowf(op),
                ))
            }
            ir::Arithmetic::Powi(op) => {
                instructions.push(Instruction::Powi(self.compile_binary(op, out)))
            }
            ir::Arithmetic::Hypot(op) => {
                let instruction = Instruction::Hypot(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Rhypot(op) => {
                let instruction = Instruction::Rhypot(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Sqrt(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Sqrt(op),
                    Instruction::FastSqrt(op),
                ))
            }
            ir::Arithmetic::InverseSqrt(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::InverseSqrt(op),
                    Instruction::FastInverseSqrt(op),
                ))
            }
            ir::Arithmetic::Erf(op) => {
                let instruction = Instruction::Erf(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Max(op) => {
                let instruction = Instruction::Max(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Min(op) => {
                let instruction = Instruction::Min(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            ir::Arithmetic::Clamp(op) => instructions.push(Instruction::Clamp {
                input: self.compile_value(op.input),
                min_value: self.compile_value(op.min_value),
                max_value: self.compile_value(op.max_value),
                out: self.compile_value(out),
            }),
            ir::Arithmetic::Recip(op) => {
                let elem = op.input.ty.elem_type();
                let input = self.compile_value(op.input);
                let out = self.compile_value(out);
                let lhs = match elem {
                    ir::ElemType::Float(_) => ir::ConstantValue::Float(1.0),
                    ir::ElemType::Int(_) => ir::ConstantValue::Int(1),
                    ir::ElemType::UInt(_) => ir::ConstantValue::UInt(1),
                    ir::ElemType::Bool => ir::ConstantValue::Bool(true),
                };
                let div = Instruction::Div(BinaryInstruction {
                    lhs: Value::Constant(lhs, self.compile_type(op.input.ty)),
                    rhs: input,
                    out,
                });
                let recip = Instruction::FastRecip(UnaryInstruction { input, out });

                instructions.push(self.select_fast_float(
                    elem.into(),
                    modes,
                    FastMath::AllowReciprocal
                        | FastMath::ReducedPrecision
                        | FastMath::UnsignedZero
                        | FastMath::NotInf,
                    div,
                    recip,
                ))
            }
            ir::Arithmetic::Round(op) => {
                instructions.push(Instruction::Round(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Floor(op) => {
                instructions.push(Instruction::Floor(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Ceil(op) => {
                instructions.push(Instruction::Ceil(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Trunc(op) => {
                instructions.push(Instruction::Trunc(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Rem(op) => {
                instructions.push(Instruction::Rem(self.compile_binary(op, out)))
            }
            ir::Arithmetic::ModFloor(op) => {
                instructions.push(Instruction::ModFloor(self.compile_binary(op, out)));
            }
            ir::Arithmetic::Fma(op) => instructions.push(Instruction::Fma {
                a: self.compile_value(op.a),
                b: self.compile_value(op.b),
                c: self.compile_value(op.c),
                out: self.compile_value(out),
            }),
            ir::Arithmetic::Neg(op) => {
                instructions.push(Instruction::Neg(self.compile_unary(op, out)))
            }
            ir::Arithmetic::Normalize(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Normalize(op),
                    Instruction::FastNormalize(op),
                ))
            }
            ir::Arithmetic::Magnitude(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Magnitude(op),
                    Instruction::FastMagnitude(op),
                ))
            }
            ir::Arithmetic::Dot(op) => {
                instructions.push(Instruction::Dot(self.compile_binary(op, out)))
            }
            ir::Arithmetic::VectorSum(op) => {
                instructions.push(Instruction::VectorSum(self.compile_unary(op, out)))
            }
        };
    }

    fn select_fast_float(
        &self,
        ty: ir::Type,
        modes: InstructionModes,
        required_flags: EnumSet<FastMath>,
        default: Instruction<D>,
        fast: Instruction<D>,
    ) -> Instruction<D> {
        if !self.compilation_options.supports_features.fast_math
            || !matches!(ty.elem_type(), ElemType::Float(FloatKind::F32))
        {
            return default;
        }

        if modes.fp_math_mode.is_superset(required_flags) {
            fast
        } else {
            default
        }
    }

    fn compile_comparison(
        &mut self,
        value: ir::Comparison,
        out: Option<ir::Value>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            ir::Comparison::Equal(op) => {
                instructions.push(Instruction::Equal(self.compile_binary(op, out)))
            }
            ir::Comparison::Lower(op) => {
                instructions.push(Instruction::Lower(self.compile_binary(op, out)))
            }
            ir::Comparison::Greater(op) => {
                instructions.push(Instruction::Greater(self.compile_binary(op, out)))
            }
            ir::Comparison::LowerEqual(op) => {
                instructions.push(Instruction::LowerEqual(self.compile_binary(op, out)))
            }
            ir::Comparison::GreaterEqual(op) => {
                instructions.push(Instruction::GreaterEqual(self.compile_binary(op, out)))
            }
            ir::Comparison::NotEqual(op) => {
                instructions.push(Instruction::NotEqual(self.compile_binary(op, out)))
            }
            ir::Comparison::IsNan(op) => {
                instructions.push(Instruction::IsNan(self.compile_unary(op, out)))
            }
            ir::Comparison::IsInf(op) => {
                instructions.push(Instruction::IsInf(self.compile_unary(op, out)))
            }
        };
    }

    fn compile_bitwise(
        &mut self,
        value: ir::Bitwise,
        out: Option<ir::Value>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            ir::Bitwise::BitwiseOr(op) => {
                instructions.push(Instruction::BitwiseOr(self.compile_binary(op, out)))
            }
            ir::Bitwise::BitwiseAnd(op) => {
                instructions.push(Instruction::BitwiseAnd(self.compile_binary(op, out)))
            }
            ir::Bitwise::BitwiseXor(op) => {
                instructions.push(Instruction::BitwiseXor(self.compile_binary(op, out)))
            }
            ir::Bitwise::CountOnes(op) => {
                instructions.push(Instruction::CountBits(self.compile_unary(op, out)))
            }
            ir::Bitwise::ReverseBits(op) => {
                instructions.push(Instruction::ReverseBits(self.compile_unary(op, out)))
            }
            ir::Bitwise::ShiftLeft(op) => {
                instructions.push(Instruction::ShiftLeft(self.compile_binary(op, out)))
            }
            ir::Bitwise::ShiftRight(op) => {
                instructions.push(Instruction::ShiftRight(self.compile_binary(op, out)))
            }
            ir::Bitwise::BitwiseNot(op) => {
                instructions.push(Instruction::BitwiseNot(self.compile_unary(op, out)))
            }
            ir::Bitwise::LeadingZeros(op) => {
                instructions.push(Instruction::LeadingZeros(self.compile_unary(op, out)))
            }
            ir::Bitwise::TrailingZeros(op) => {
                instructions.push(Instruction::TrailingZeros(self.compile_unary(op, out)))
            }
            ir::Bitwise::FindFirstSet(op) => {
                let instruction = Instruction::FindFirstSet(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
        };
    }

    fn compile_memory(
        &mut self,
        value: ir::Memory,
        out: Option<ir::Value>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        match value {
            ir::Memory::Index(op) => {
                instructions.push(Instruction::Index(self.compile_index(op, out.unwrap())))
            }
            ir::Memory::Load(value) => instructions.push(Instruction::Load(UnaryInstruction {
                input: self.compile_value(value),
                out: self.compile_value(out.unwrap()),
            })),
            ir::Memory::Store(op) => instructions.push(Instruction::Store(UnaryInstruction {
                input: self.compile_value(op.value),
                out: self.compile_value(op.ptr),
            })),
            ir::Memory::CopyMemory(op) => instructions.push(Instruction::Copy {
                source: self.compile_value(op.source),
                dest: self.compile_value(op.target),
                len: op.len as u32,
            }),
        };
    }

    fn compile_operator(
        &mut self,
        value: ir::Operator,
        out: Option<ir::Value>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            ir::Operator::And(op) => {
                instructions.push(Instruction::And(self.compile_binary(op, out)))
            }
            ir::Operator::Or(op) => {
                instructions.push(Instruction::Or(self.compile_binary(op, out)))
            }
            ir::Operator::Not(op) => {
                instructions.push(Instruction::Not(self.compile_unary(op, out)))
            }
            ir::Operator::InitVector(op) => instructions.push(Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|it| self.compile_value(it))
                    .collect(),
                out: self.compile_value(out),
            }),
            ir::Operator::InsertComponent(op) => instructions.push(Instruction::InsertComponent {
                vector: self.compile_value(op.vector),
                index: self.compile_value(op.index),
                value: self.compile_value(op.value),
                out: self.compile_value(out),
            }),
            ir::Operator::ExtractComponent(op) => {
                instructions.push(Instruction::ExtractComponent(self.compile_binary(op, out)))
            }
            ir::Operator::Select(op) => instructions.push(Instruction::Select {
                cond: self.compile_value(op.cond),
                then: self.compile_value(op.then),
                or_else: self.compile_value(op.or_else),
                out: self.compile_value(out),
            }),
            // Needs special conversion semantics
            ir::Operator::Cast(op)
                if (is_fp4_fp6_fp8(op.input.elem_type()) || is_fp4_fp6_fp8(out.elem_type()))
                // Trivial broadcast shouldn't use special cast logic
                    && op.input.elem_type() != out.elem_type() =>
            {
                // We may need these for intermediates
                self.flags.elem_f16 = true;
                self.flags.elem_bf16 = true;
                let vec_in = op.input.ty.vector_size();
                let packing = out.storage_type().packing_factor();
                self.compile_type(op.input.ty.with_vector_size(packing));
                self.compile_type(
                    ir::Type::scalar(ir::ElemType::Float(FloatKind::F16)).with_vector_size(vec_in),
                );
                self.compile_type(
                    ir::Type::scalar(ir::ElemType::Float(FloatKind::BF16)).with_vector_size(vec_in),
                );
                self.compile_type(
                    ir::Type::scalar(ir::ElemType::Float(FloatKind::F16)).with_vector_size(packing),
                );
                self.compile_type(
                    ir::Type::scalar(ir::ElemType::Float(FloatKind::BF16))
                        .with_vector_size(packing),
                );

                let inst = self.compile_unary(op, out);

                instructions.push(Instruction::SpecialCast(inst));
            }
            ir::Operator::Cast(op) => {
                let op = self.compile_unary(op, out);

                if op.input.elem() == Elem::TF32 || op.out.elem() == Elem::TF32 {
                    self.flags.elem_tf32 = true;
                }

                instructions.push(Instruction::Assign(op))
            }
            ir::Operator::Reinterpret(op) => {
                instructions.push(Instruction::Bitcast(self.compile_unary(op, out)))
            }
            ir::Operator::ReadBuiltin(builtin) => {
                let out = self.compile_value(out);
                let mut assign = |input| {
                    instructions.push(Instruction::Assign(UnaryInstruction { input, out }));
                };
                let builtin = match builtin {
                    ir::Builtin::AbsolutePos => {
                        self.flags.indexes.absolute_pos = true;
                        Builtin::AbsolutePos(*self.addr_type.elem())
                    }
                    ir::Builtin::CubePosCluster
                        if self.compilation_options.supports_features.clusters =>
                    {
                        self.flags.indexes.cluster_pos = true;
                        Builtin::ClusterRank
                    }
                    ir::Builtin::CubePosClusterX
                        if self.compilation_options.supports_features.clusters =>
                    {
                        self.flags.indexes.cluster_pos = true;
                        Builtin::ClusterIndexX
                    }
                    ir::Builtin::CubePosClusterY
                        if self.compilation_options.supports_features.clusters =>
                    {
                        self.flags.indexes.cluster_pos = true;
                        Builtin::ClusterIndexY
                    }
                    ir::Builtin::CubePosClusterZ
                        if self.compilation_options.supports_features.clusters =>
                    {
                        self.flags.indexes.cluster_pos = true;
                        Builtin::ClusterIndexZ
                    }
                    // Fallback if clusters aren't supported, ID is always 0 since clusters are always
                    // (1, 1, 1) if unsupported
                    ir::Builtin::CubePosCluster
                    | ir::Builtin::CubePosClusterX
                    | ir::Builtin::CubePosClusterY
                    | ir::Builtin::CubePosClusterZ => {
                        assign(const_u32(0));
                        return;
                    }
                    ir::Builtin::AbsolutePosX => {
                        self.flags.indexes.absolute_pos_tuple = true;
                        Builtin::AbsolutePosX
                    }
                    ir::Builtin::AbsolutePosY => {
                        self.flags.indexes.absolute_pos_tuple = true;
                        Builtin::AbsolutePosY
                    }
                    ir::Builtin::AbsolutePosZ => {
                        self.flags.indexes.absolute_pos_tuple = true;
                        Builtin::AbsolutePosZ
                    }
                    ir::Builtin::CubeDim => {
                        self.flags.indexes.cube_dim = true;
                        Builtin::CubeDim
                    }
                    ir::Builtin::CubeDimX => {
                        self.flags.indexes.cube_dim_tuple = true;
                        Builtin::CubeDimX
                    }
                    ir::Builtin::CubeDimY => {
                        self.flags.indexes.cube_dim_tuple = true;
                        Builtin::CubeDimY
                    }
                    ir::Builtin::CubeDimZ => {
                        self.flags.indexes.cube_dim_tuple = true;
                        Builtin::CubeDimZ
                    }
                    ir::Builtin::CubeClusterDim => {
                        assign(const_u32(self.cluster_dim.num_elems()));
                        return;
                    }
                    ir::Builtin::CubeClusterDimX => {
                        assign(const_u32(self.cluster_dim.x));
                        return;
                    }
                    ir::Builtin::CubeClusterDimY => {
                        assign(const_u32(self.cluster_dim.y));
                        return;
                    }
                    ir::Builtin::CubeClusterDimZ => {
                        assign(const_u32(self.cluster_dim.z));
                        return;
                    }
                    ir::Builtin::CubePos => {
                        self.flags.indexes.cube_pos = true;
                        Builtin::CubePos(*self.addr_type.elem())
                    }
                    ir::Builtin::CubePosX => {
                        self.flags.indexes.cube_pos_tuple = true;
                        Builtin::CubePosX
                    }
                    ir::Builtin::CubePosY => {
                        self.flags.indexes.cube_pos_tuple = true;
                        Builtin::CubePosY
                    }
                    ir::Builtin::CubePosZ => {
                        self.flags.indexes.cube_pos_tuple = true;
                        Builtin::CubePosZ
                    }
                    ir::Builtin::CubeCount => {
                        self.flags.indexes.cube_count = true;
                        Builtin::CubeCount(*self.addr_type.elem())
                    }
                    ir::Builtin::CubeCountX => {
                        self.flags.indexes.cube_count_tuple = true;
                        Builtin::CubeCountX
                    }
                    ir::Builtin::CubeCountY => {
                        self.flags.indexes.cube_count_tuple = true;
                        Builtin::CubeCountY
                    }
                    ir::Builtin::CubeCountZ => {
                        self.flags.indexes.cube_count_tuple = true;
                        Builtin::CubeCountZ
                    }
                    ir::Builtin::UnitPos => {
                        self.flags.indexes.unit_pos = true;
                        Builtin::UnitPos
                    }
                    ir::Builtin::UnitPosX => {
                        self.flags.indexes.unit_pos_tuple = true;
                        Builtin::UnitPosX
                    }
                    ir::Builtin::UnitPosY => {
                        self.flags.indexes.unit_pos_tuple = true;
                        Builtin::UnitPosY
                    }
                    ir::Builtin::UnitPosZ => {
                        self.flags.indexes.unit_pos_tuple = true;
                        Builtin::UnitPosZ
                    }
                    ir::Builtin::PlaneDim => {
                        self.flags.indexes.plane_dim = true;
                        Builtin::PlaneDim
                    }
                    ir::Builtin::PlanePos => {
                        self.flags.indexes.plane_pos = true;
                        Builtin::PlanePos
                    }
                    ir::Builtin::UnitPosPlane => {
                        self.flags.indexes.unit_pos_plane = true;
                        Builtin::UnitPosPlane
                    }
                };
                instructions.push(Instruction::ReadBuiltin { builtin, out })
            }
            ir::Operator::ReadScalar(id) => instructions.push(Instruction::ReadScalar {
                id,
                out: self.compile_value(out),
            }),
        };
    }

    fn compile_binary(
        &mut self,
        value: ir::BinaryOperands,
        out: ir::Value,
    ) -> BinaryInstruction<D> {
        BinaryInstruction {
            lhs: self.compile_value(value.lhs),
            rhs: self.compile_value(value.rhs),
            out: self.compile_value(out),
        }
    }

    fn compile_atomic_binary(
        &mut self,
        value: ir::AtomicBinaryOperands,
        out: ir::Value,
    ) -> BinaryInstruction<D> {
        BinaryInstruction {
            lhs: self.compile_value(value.ptr),
            rhs: self.compile_value(value.value),
            out: self.compile_value(out),
        }
    }

    fn compile_index(&mut self, value: ir::IndexOperands, out: ir::Value) -> IndexInstruction<D> {
        IndexInstruction {
            list: self.compile_value(value.list),
            index: self.compile_value(value.index),
            out: self.compile_value(out),
        }
    }

    fn compile_unary(&mut self, value: ir::UnaryOperands, out: ir::Value) -> UnaryInstruction<D> {
        UnaryInstruction {
            input: self.compile_value(value.input),
            out: self.compile_value(out),
        }
    }

    fn compile_value(&mut self, value: ir::Value) -> Value<D> {
        let item = value.ty;
        match value.kind {
            ir::ValueKind::Value { id } => Value::Value {
                id,
                item: self.compile_type(item),
            },
            ir::ValueKind::Constant(value) => Value::Constant(value, self.compile_type(item)),
        }
    }

    fn compile_matrix(&mut self, matrix: ir::MatrixType) -> FragmentType<D> {
        FragmentType {
            ident: self.compile_matrix_ident(matrix.ident),
            m: matrix.m as u32,
            n: matrix.n as u32,
            k: matrix.k as u32,
            elem: self.compile_storage_type(matrix.storage),
            layout: self.compile_matrix_layout(matrix.layout),
        }
    }

    fn compile_matrix_ident(&mut self, ident: ir::MatrixIdent) -> FragmentIdent<D> {
        match ident {
            ir::MatrixIdent::A => FragmentIdent::A,
            ir::MatrixIdent::B => FragmentIdent::B,
            ir::MatrixIdent::Accumulator => FragmentIdent::Accumulator,
        }
    }

    fn compile_matrix_layout(&mut self, layout: ir::MatrixLayout) -> Option<FragmentLayout<D>> {
        match layout {
            ir::MatrixLayout::ColMajor => Some(FragmentLayout::ColMajor),
            ir::MatrixLayout::RowMajor => Some(FragmentLayout::RowMajor),
            ir::MatrixLayout::Undefined => None,
        }
    }

    fn compile_binding(&mut self, binding: cubecl_runtime::kernel::KernelArg) -> KernelArg<D> {
        KernelArg {
            id: binding.id,
            value: self.compile_value(binding.value),
            vis: self.buffer_vis[binding.id as usize],
        }
    }

    fn compile_type(&mut self, ty: ir::Type) -> Item<D> {
        let item = match ty {
            ir::Type::Scalar(ty) => Item::Scalar(self.compile_storage_type(ty)),
            ir::Type::Vector(ty, vector_size) => {
                Item::Vector(self.compile_type(*ty).intern(), vector_size)
            }
            ir::Type::Atomic(ty) => {
                let item = self.compile_type(*ty);
                Item::Atomic(item.intern())
            }
            ir::Type::Pointer(ty, class) => {
                let item = self.compile_type(*ty);
                let class = match class {
                    ir::AddressSpace::Global(id) => {
                        PointerClass::Global(self.buffer_vis[id as usize])
                    }
                    ir::AddressSpace::Shared => PointerClass::Shared,
                    ir::AddressSpace::Local => PointerClass::Local,
                };
                Item::Pointer(item.intern(), class)
            }
            ir::Type::Array(ty, size) => {
                let ty = self.compile_type(*ty);
                Item::Array(ty.intern(), size)
            }
            ir::Type::DynamicArray(ty) => {
                let ty = self.compile_type(*ty);
                Item::DynamicArray(ty.intern())
            }
            ir::Type::Matrix(ty) => {
                let ty = self.compile_matrix(ty);
                Item::Fragment(ty)
            }
            ir::Type::Semantic(ty) => self.compile_semantic_type(ty),
            ir::Type::Opaque(ty) => self.compile_opaque_type(ty),
            ir::Type::Aggregate(_) => {
                unreachable!("Should be disaggregated at this point")
            }
        };
        if *item.elem() != super::Elem::TF32 {
            self.items.insert(item);
            self.items.insert(item.optimized());
        } else {
            // TF32 is represented as `float` in C++
            let item = item.with_elem(super::Elem::F32);
            self.items.insert(item);
        }

        item
    }

    fn compile_storage_type(&mut self, value: ir::StorageType) -> Elem<D> {
        match value {
            ir::StorageType::Scalar(ty) => self.compile_elem(ty),
            ir::StorageType::Packed(ir::ElemType::Float(kind), 2) => match kind {
                FloatKind::E2M1 => {
                    self.flags.elem_fp4 = true;
                    Elem::FP4x2(FP4Kind::E2M1)
                }
                FloatKind::E2M3 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6x2(FP6Kind::E2M3)
                }
                FloatKind::E3M2 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E3M2)
                }
                FloatKind::E4M3 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8x2(FP8Kind::E4M3)
                }
                FloatKind::E5M2 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8x2(FP8Kind::E5M2)
                }
                FloatKind::UE8M0 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8x2(FP8Kind::UE8M0)
                }
                FloatKind::F16 => {
                    self.flags.elem_f16 = true;
                    Elem::F16x2
                }
                FloatKind::BF16 => {
                    self.flags.elem_bf16 = true;
                    Elem::BF16x2
                }
                other => unimplemented!("Unsupported storage type: packed<{other:?}, 2>"),
            },
            ir::StorageType::Packed(other, factor) => {
                unimplemented!("Unsupported storage type: packed<{other}, {factor}>")
            }
        }
    }

    fn compile_elem(&mut self, value: ir::ElemType) -> Elem<D> {
        match value {
            ir::ElemType::Float(kind) => match kind {
                ir::FloatKind::E2M1 => {
                    self.flags.elem_fp4 = true;
                    Elem::FP4(FP4Kind::E2M1)
                }
                ir::FloatKind::E2M3 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E2M3)
                }
                ir::FloatKind::E3M2 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E3M2)
                }
                ir::FloatKind::E4M3 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::E4M3)
                }
                ir::FloatKind::E5M2 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::E5M2)
                }
                ir::FloatKind::UE8M0 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::UE8M0)
                }
                ir::FloatKind::F16 => {
                    self.flags.elem_f16 = true;
                    Elem::F16
                }
                ir::FloatKind::BF16 => {
                    self.flags.elem_bf16 = true;
                    Elem::BF16
                }
                ir::FloatKind::TF32 => Elem::TF32,
                ir::FloatKind::Flex32 => Elem::F32,
                ir::FloatKind::F32 => Elem::F32,
                ir::FloatKind::F64 => Elem::F64,
            },
            ir::ElemType::Int(kind) => match kind {
                ir::IntKind::I8 => Elem::I8,
                ir::IntKind::I16 => Elem::I16,
                ir::IntKind::I32 => Elem::I32,
                ir::IntKind::I64 => Elem::I64,
            },
            ir::ElemType::UInt(kind) => match kind {
                ir::UIntKind::U8 => Elem::U8,
                ir::UIntKind::U16 => Elem::U16,
                ir::UIntKind::U32 => Elem::U32,
                ir::UIntKind::U64 => Elem::U64,
            },
            ir::ElemType::Bool => Elem::Bool,
        }
    }

    fn compile_semantic_type(&mut self, value: ir::SemanticType) -> Item<D> {
        match value {
            ir::SemanticType::TensorLayout(..) | ir::SemanticType::TensorView(..) => {
                panic!("Tensor addressing is only supported on Vulkan")
            }
        }
    }

    fn compile_opaque_type(&mut self, value: ir::OpaqueType) -> Item<D> {
        match value {
            ir::OpaqueType::Barrier(barrier_level) => {
                self.flags.op_barrier = true;
                Item::Barrier(barrier_level)
            }
            ir::OpaqueType::BarrierToken(barrier_level) => {
                self.flags.op_barrier = true;
                Item::BarrierToken(barrier_level)
            }
            ir::OpaqueType::TensorMap => Item::TensorMap,
        }
    }
}

fn is_fp4_fp6_fp8(elem: ir::ElemType) -> bool {
    match elem {
        ir::ElemType::Float(kind) => matches!(
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

fn const_u32<D: Dialect>(value: u32) -> Value<D> {
    Value::Constant(
        ir::ConstantValue::UInt(value as u64),
        Item::Scalar(Elem::U32),
    )
}

pub fn register_supported_types(props: &mut DeviceProperties) {
    props.register_address_type(ir::AddressType::U32);
    props.register_address_type(ir::AddressType::U64);

    let supported_types = [
        ir::ElemType::UInt(ir::UIntKind::U8),
        ir::ElemType::UInt(ir::UIntKind::U16),
        ir::ElemType::UInt(ir::UIntKind::U32),
        ir::ElemType::UInt(ir::UIntKind::U64),
        ir::ElemType::Int(ir::IntKind::I8),
        ir::ElemType::Int(ir::IntKind::I16),
        ir::ElemType::Int(ir::IntKind::I32),
        ir::ElemType::Int(ir::IntKind::I64),
        ir::ElemType::Float(ir::FloatKind::BF16),
        ir::ElemType::Float(ir::FloatKind::F16),
        ir::ElemType::Float(ir::FloatKind::F32),
        ir::ElemType::Float(ir::FloatKind::Flex32),
        ir::ElemType::Float(ir::FloatKind::F64),
        ir::ElemType::Bool,
    ];

    let supported_atomic_types = [
        ir::ElemType::Int(ir::IntKind::I32),
        ir::ElemType::Int(ir::IntKind::I64),
        ir::ElemType::UInt(ir::UIntKind::U32),
        ir::ElemType::UInt(ir::UIntKind::U64),
        ir::ElemType::Float(ir::FloatKind::F32),
    ];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in supported_atomic_types {
        props.register_atomic_type_usage(
            Type::atomic(ty),
            AtomicUsage::Add | AtomicUsage::LoadStore,
        );
    }
}
