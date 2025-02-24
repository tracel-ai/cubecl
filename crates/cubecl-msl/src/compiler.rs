use cubecl_common::ExecutionMode;
use cubecl_core::{
    compute::{self, Visibility},
    ir::{self as cube, Scope},
    prelude::{expand_checked_index_assign, expand_erf},
    Metadata, WgpuCompilationOptions as MslCompilationOptions,
};

use crate::{
    AddressSpace, Binding, Body, ConstantArray, Elem, Instruction, Item, LocalArray, MetalKernel, SharedMemory, Subgroup, Variable
};

/// Metal Compiler.
#[derive(Clone, Default)]
pub struct MslCompiler {
    // thread position in grid
    thread_index_in_grid: bool,
    thread_position_in_grid: bool,
    // thread count in threadgroup
    total_threads_in_threadgroup: bool,
    threads_per_threadgroup: bool,
    // thread position in threadgroup
    thread_index_in_threadgroup: bool,
    thread_position_in_threadgroup: bool,
    // threadgroup count in grid
    total_threadgroups_in_grid: bool,
    threadgroups_per_grid: bool,
    // threadgroup position in grid
    pub threadgroup_index_in_grid: bool,
    pub threadgroup_position_in_grid: bool,
    // simd-groups
    pub thread_index_in_simdgroup: bool,
    pub threads_per_simdgroup: bool,


    num_inputs: usize,
    num_outputs: usize,
    metadata: Metadata,
    ext_meta_pos: Vec<u32>,
    shared_memories: Vec<SharedMemory>,
    const_arrays: Vec<ConstantArray>,
    local_arrays: Vec<LocalArray>,
    #[allow(dead_code)]
    compilation_options: MslCompilationOptions,
    strategy: ExecutionMode,
    subgroup_instructions_used: bool,
}

impl core::fmt::Debug for MslCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MetalCompiler")
    }
}

impl cubecl_core::Compiler for MslCompiler {
    type Representation = MetalKernel;
    type CompilationOptions = MslCompilationOptions;

    fn compile(
        &mut self,
        shader: compute::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        self.compilation_options = compilation_options.clone();
        self.compile_kernel(shader, mode)
    }

    fn elem_size(&self, elem: cube::Elem) -> usize {
        Self::compile_elem(elem).size()
    }
}

impl MslCompiler {
    fn compile_kernel(
        &mut self,
        mut value: compute::KernelDefinition,
        mode: ExecutionMode,
    ) -> MetalKernel {
        self.strategy = mode;

        self.num_inputs = value.inputs.len();
        self.num_outputs = value.outputs.len();
        let num_meta = value.inputs.len() + value.outputs.len();

        self.ext_meta_pos = Vec::new();
        let mut num_ext = 0;

        for binding in value.inputs.iter().chain(value.outputs.iter()) {
            self.ext_meta_pos.push(num_ext);
            if binding.has_extended_meta {
                num_ext += 1;
            }
        }

        self.metadata = Metadata::new(num_meta as u32, num_ext);

        let instructions = self.compile_scope(&mut value.body);
        let body = Body {
            instructions,
        };

        // builtins inclusion rules
        let thread_index_in_grid = self.thread_index_in_grid;
        let thread_position_in_grid = self.thread_position_in_grid || thread_index_in_grid;
        let total_threads_in_threadgroup = self.total_threads_in_threadgroup;
        let threads_per_threadgroup = self.threads_per_threadgroup || total_threads_in_threadgroup || thread_index_in_grid;
        let thread_index_in_threadgroup = self.thread_index_in_threadgroup;
        let thread_position_in_threadgroup = self.thread_position_in_threadgroup || thread_index_in_threadgroup;
        let total_threadgroups_in_grid = self.total_threadgroups_in_grid;
        let threadgroups_per_grid = self.threadgroups_per_grid || thread_index_in_grid;
        let threadgroup_index_in_grid = self.threadgroup_index_in_grid;
        let threadgroup_position_in_grid = self.threadgroup_position_in_grid || threadgroup_index_in_grid;
        let threads_per_simdgroup = self.threads_per_simdgroup;
        let thread_index_in_simdgroup = self.thread_index_in_simdgroup;

        MetalKernel {
            cube_dim: value.cube_dim,
            inputs: value
                .inputs
                .into_iter()
                .map(Self::compile_binding)
                .collect(),
            outputs: value
                .outputs
                .into_iter()
                .map(Self::compile_binding)
                .collect(),
            named: value
                .named
                .into_iter()
                .map(|(name, b)| (name, Self::compile_binding(b)))
                .collect(),
            thread_index_in_grid,
            thread_position_in_grid,
            total_threads_in_threadgroup,
            threads_per_threadgroup,
            thread_index_in_threadgroup,
            thread_position_in_threadgroup,
            total_threadgroups_in_grid,
            threadgroups_per_grid,
            threadgroup_index_in_grid,
            threadgroup_position_in_grid,
            thread_index_in_simdgroup,
            threads_per_simdgroup,


            shared_memories: self.shared_memories.clone(),
            constant_arrays: self.const_arrays.clone(),
            local_arrays: self.local_arrays.clone(),
            body,
            kernel_name: value.options.kernel_name,
        }
    }

    fn compile_item(item: cube::Item) -> Item {
        let elem = Self::compile_elem(item.elem);
        match item.vectorization.map(|it| it.get()).unwrap_or(1) {
            1 => Item::Scalar(elem),
            2 => Item::Vec2(elem),
            3 => Item::Vec3(elem),
            4 => Item::Vec4(elem),
            _ => panic!("Unsupported vectorizations scheme {:?}", item.vectorization),
        }
    }

    fn compile_elem(value: cube::Elem) -> Elem {
        match value {
            cube::Elem::AtomicFloat(kind) => match kind {
                cube::FloatKind::F32 => Elem::Atomic(crate::AtomicKind::F32),
                kind => panic!("atomic_{kind:?} is not a valid Metal data type"),
            },
            cube::Elem::AtomicInt(kind) => match kind {
                cube::IntKind::I32 => Elem::Atomic(crate::AtomicKind::I32),
                cube::IntKind::I64 => Elem::Atomic(crate::AtomicKind::I64),
                kind => panic!("atomic_{kind:?} is not a valid Metal data type"),
            },
            cube::Elem::AtomicUInt(kind) => match kind {
                cube::UIntKind::U32 => Elem::Atomic(crate::AtomicKind::U32),
                cube::UIntKind::U64 => Elem::Atomic(crate::AtomicKind::U64),
                kind => panic!("atomic_{kind:?} is not a valid Metal data type"),
            },
            cube::Elem::Bool => Elem::Bool,
            cube::Elem::Int(i) => match i {
                cube::IntKind::I32 => Elem::I32,
                kind => panic!("{kind:?} is not a valid MetalElement"),
            },
            cube::Elem::UInt(kind) => match kind {
                cube::UIntKind::U32 => Elem::U32,
                kind => panic!("{kind:?} is not a valid MetalElement"),
            },
            cube::Elem::Float(f) => match f {
                cube::FloatKind::F16 => Elem::F16,
                cube::FloatKind::BF16 => Elem::BF16,
                cube::FloatKind::TF32 => panic!("tf32 is not a valid Metal data type"),
                cube::FloatKind::Flex32 => Elem::F32,
                cube::FloatKind::F32 => Elem::F32,
                cube::FloatKind::F64 => panic!("f64 is not a valid Metal data type"),
            },
        }
    }

    fn ext_meta_pos(&self, var: &cube::Variable) -> u32 {
        let pos = match var.kind {
            cube::VariableKind::GlobalInputArray(id) => id as usize,
            cube::VariableKind::GlobalOutputArray(id) => self.num_inputs + id as usize,
            _ => panic!("Only global arrays have metadata"),
        };
        self.ext_meta_pos[pos]
    }

    pub(crate) fn compile_variable(&mut self, value: cube::Variable) -> Variable {
        let item = value.item;
        match value.kind {
            cube::VariableKind::Builtin(builtin) => match builtin {
                // thread position in grid
                cube::Builtin::AbsolutePos => {
                    self.thread_index_in_grid = true;
                    Variable::ThreadIndexInGrid
                }
                cube::Builtin::AbsolutePosX => {
                    self.thread_position_in_grid = true;
                    Variable::ThreadPositionInGridX
                }
                cube::Builtin::AbsolutePosY => {
                    self.thread_position_in_grid = true;
                    Variable::ThreadPositionInGridY
                }
                cube::Builtin::AbsolutePosZ => {
                    self.thread_position_in_grid = true;
                    Variable::ThreadPositionInGridZ
                }
                // thread count in threadgroup
                cube::Builtin::CubeDim => {
                    self.total_threads_in_threadgroup = true;
                    Variable::TotalThreadsInThreadgroup
                }
                cube::Builtin::CubeDimX => {
                    self.threads_per_threadgroup = true;
                    Variable::ThreadsPerThreadgoupX
                },
                cube::Builtin::CubeDimY => {
                    self.threads_per_threadgroup = true;
                    Variable::ThreadsPerThreadgoupY
                },
                cube::Builtin::CubeDimZ => {
                    self.threads_per_threadgroup = true;
                    Variable::ThreadsPerThreadgoupZ
                },
                // thread position in threadgroup
                cube::Builtin::UnitPos => {
                    self.thread_index_in_threadgroup = true;
                    Variable::ThreadIndexInThreadgroup
                }
                cube::Builtin::UnitPosX => {
                    self.thread_position_in_threadgroup = true;
                    Variable::ThreadPositionInThreadgroupX
                }
                cube::Builtin::UnitPosY => {
                    self.thread_position_in_threadgroup = true;
                    Variable::ThreadPositionInThreadgroupY
                }
                cube::Builtin::UnitPosZ => {
                    self.thread_position_in_threadgroup = true;
                    Variable::ThreadPositionInThreadgroupZ
                }
                // threadgroup count in grid
                cube::Builtin::CubeCount => {
                    self.total_threadgroups_in_grid = true;
                    Variable::TotalThreadgroupsInGrid
                }
                cube::Builtin::CubeCountX => {
                    self.threadgroups_per_grid = true;
                    Variable::ThreadgroupsPerGridX
                }
                cube::Builtin::CubeCountY => {
                    self.threadgroups_per_grid = true;
                    Variable::ThreadgroupsPerGridY
                }
                cube::Builtin::CubeCountZ => {
                    self.threadgroups_per_grid = true;
                    Variable::ThreadgroupsPerGridZ
                }
                // threadgroup position in grid
                cube::Builtin::CubePos => {
                    self.threadgroup_index_in_grid = true;
                    Variable::ThreadgroupIndexInGrid
                }
                cube::Builtin::CubePosX => {
                    self.threadgroup_position_in_grid = true;
                    Variable::ThreadgroupPositionInGridX
                }
                cube::Builtin::CubePosY => {
                    self.threadgroup_position_in_grid = true;
                    Variable::ThreadgroupPositionInGridY
                }
                cube::Builtin::CubePosZ => {
                    self.threadgroup_position_in_grid = true;
                    Variable::ThreadgroupPositionInGridZ
                }
                // simd-groups
                cube::Builtin::UnitPosPlane => {
                    self.thread_index_in_simdgroup = true;
                    Variable::ThreadIndexInSIMDgroup
                }
                cube::Builtin::PlaneDim => {
                    self.threads_per_simdgroup = true;
                    Variable::ThreadsPerSIMDgroup
                }
            },
            cube::VariableKind::GlobalInputArray(id) => {
                Variable::GlobalInputArray(id, Self::compile_item(item))
            }
            cube::VariableKind::GlobalOutputArray(id) => {
                Variable::GlobalOutputArray(id, Self::compile_item(item))
            }
            cube::VariableKind::GlobalScalar(id) => {
                Variable::GlobalScalar(id, Self::compile_elem(item.elem), item.elem)
            }




            cube::VariableKind::LocalMut { id } | cube::VariableKind::Versioned { id, .. } => {
                Variable::LocalMut {
                    id,
                    item: Self::compile_item(item),
                }
            }
            cube::VariableKind::LocalConst { id } => Variable::LocalConst {
                id,
                item: Self::compile_item(item),
            },
            cube::VariableKind::Slice { id } => Variable::Slice {
                id,
                item: Self::compile_item(item),
            },
            cube::VariableKind::ConstantScalar(value) => {
                Variable::ConstantScalar(value, Self::compile_elem(value.elem()))
            }
            cube::VariableKind::SharedMemory { id, length } => {
                let item = Self::compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == id) {
                    self.shared_memories
                        .push(SharedMemory::new(id, item, length));
                }
                Variable::SharedMemory(id, item, length)
            }
            cube::VariableKind::ConstantArray { id, length } => {
                let item = Self::compile_item(item);
                Variable::ConstantArray(id, item, length)
            }
            cube::VariableKind::LocalArray { id, length } => {
                let item = Self::compile_item(item);
                if !self.local_arrays.iter().any(|s| s.index == id) {
                    self.local_arrays.push(LocalArray::new(id, item, length));
                }
                Variable::LocalArray(id, item, length)
            }
            cube::VariableKind::Matrix { .. } => {
                panic!("Cooperative matrix-multiply and accumulate not supported.")
            }
            cube::VariableKind::Pipeline { .. } => {
                panic!("Pipeline not supported.")
            }
            cube::VariableKind::Barrier { .. } => {
                panic!("Barrier not supported.")
            },
        }
    }

    fn compile_scope(&mut self, scope: &mut cube::Scope) -> Vec<Instruction> {
        let mut instructions = Vec::new();

        let const_arrays = scope
            .const_arrays
            .drain(..)
            .map(|(var, values)| ConstantArray {
                index: var.index().unwrap(),
                item: Self::compile_item(var.item),
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
            // We don't declare slices.
            if let cube::VariableKind::Slice { .. } = var.kind {
                continue;
            }

            instructions.push(Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .instructions
            .into_iter()
            .for_each(|op| self.compile_operation(&mut instructions, op.operation, op.out, scope));

        instructions
    }

    fn compile_operation(
        &mut self,
        instructions: &mut Vec<Instruction>,
        operation: cube::Operation,
        out: Option<cube::Variable>,
        scope: &mut cube::Scope,
    ) {
        match operation {
            cube::Operation::Copy(variable) => instructions.push(Instruction::Assign {
                input: self.compile_variable(variable),
                out: self.compile_variable(out.unwrap()),
            }),
            cube::Operation::Arithmetic(op) => {
                self.compile_arithmetic(op, out, instructions, scope)
            }
            cube::Operation::Comparison(op) => self.compile_cmp(op, out, instructions),
            cube::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
            cube::Operation::Operator(op) => self.compile_operator(op, out, instructions, scope),
            cube::Operation::Atomic(op) => instructions.push(self.compile_atomic(op, out)),
            cube::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            cube::Operation::Branch(val) => self.compile_branch(instructions, val),
            cube::Operation::Synchronization(val) => {
                self.compile_synchronization(instructions, val)
            }
            cube::Operation::Plane(op) => self.compile_subgroup(instructions, op, out),
            cube::Operation::CoopMma(_) => {
                panic!("Cooperative matrix-multiply and accumulate isn't supported on wgpu.")
            }
            cube::Operation::NonSemantic(cube::NonSemantic::Comment { content }) => {
                self.compile_comment(instructions, content)
            }
            // No good way to attach debug info
            cube::Operation::NonSemantic(_) => {}
            cube::Operation::Pipeline(_) => {
                panic!("Pipeline isn't supported on wgpu.")
            }
            cube::Operation::Barrier(_) => {
                panic!("Barrier isn't supported on wgpu.")
            },
        }
    }

    fn compile_subgroup(
        &mut self,
        instructions: &mut Vec<Instruction>,
        subgroup: cube::Plane,
        out: Option<cube::Variable>,
    ) {
        self.subgroup_instructions_used = true;

        let out = out.unwrap();
        let op = match subgroup {
            cube::Plane::Elect => Subgroup::Elect {
                out: self.compile_variable(out),
            },
            cube::Plane::All(op) => Subgroup::All {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::Any(op) => Subgroup::Any {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::Ballot(op) => Subgroup::Ballot {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::Broadcast(op) => Subgroup::Broadcast {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::Plane::Sum(op) => Subgroup::Sum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::ExclusiveSum(op) => Subgroup::ExclusiveSum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::InclusiveSum(op) => Subgroup::InclusiveSum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::Prod(op) => Subgroup::Prod {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::ExclusiveProd(op) => Subgroup::ExclusiveProd {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::InclusiveProd(op) => Subgroup::InclusiveProd {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::Min(op) => Subgroup::Min {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::Plane::Max(op) => Subgroup::Max {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
        };

        instructions.push(Instruction::Subgroup(op));
    }

    fn compile_branch(&mut self, instructions: &mut Vec<Instruction>, branch: cube::Branch) {
        match branch {
            cube::Branch::If(mut op) => instructions.push(Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            cube::Branch::IfElse(mut op) => instructions.push(Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            cube::Branch::Switch(mut op) => instructions.push(Instruction::Switch {
                value: self.compile_variable(op.value),
                instructions_default: self.compile_scope(&mut op.scope_default),
                cases: op
                    .cases
                    .into_iter()
                    .map(|(val, mut scope)| {
                        (self.compile_variable(val), self.compile_scope(&mut scope))
                    })
                    .collect(),
            }),
            cube::Branch::Return => instructions.push(Instruction::Return),
            cube::Branch::Break => instructions.push(Instruction::Break),
            cube::Branch::RangeLoop(mut range_loop) => instructions.push(Instruction::RangeLoop {
                i: self.compile_variable(range_loop.i),
                start: self.compile_variable(range_loop.start),
                end: self.compile_variable(range_loop.end),
                step: range_loop.step.map(|it| self.compile_variable(it)),
                inclusive: range_loop.inclusive,
                instructions: self.compile_scope(&mut range_loop.scope),
            }),
            cube::Branch::Loop(mut op) => instructions.push(Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }

    fn compile_synchronization(
        &mut self,
        instructions: &mut Vec<Instruction>,
        synchronization: cube::Synchronization,
    ) {
        match synchronization {
            cube::Synchronization::SyncUnits => instructions.push(Instruction::WorkgroupBarrier),
            cube::Synchronization::SyncStorage => instructions.push(Instruction::StorageBarrier),
        };
    }

    fn compile_comment(&mut self, instructions: &mut Vec<Instruction>, content: String) {
        instructions.push(Instruction::Comment { content })
    }

    fn compile_metadata(
        &mut self,
        metadata: cube::Metadata,
        out: Option<cube::Variable>,
    ) -> Instruction {
        let out = out.unwrap();
        match metadata {
            cube::Metadata::Rank { var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.metadata.rank_index(position);
                Instruction::Metadata {
                    out: self.compile_variable(out),
                    info_offset: self.compile_variable(offset.into()),
                }
            }
            cube::Metadata::Stride { dim, var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.metadata.stride_offset_index(position);
                Instruction::ExtendedMeta {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            cube::Metadata::Shape { dim, var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.metadata.shape_offset_index(position);
                Instruction::ExtendedMeta {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            cube::Metadata::Length { var } => match var.kind {
                cube::VariableKind::GlobalInputArray(id) => {
                    let offset = self.metadata.len_index(id);
                    Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                cube::VariableKind::GlobalOutputArray(id) => {
                    let offset = self.metadata.len_index(self.num_inputs as u32 + id);
                    Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                _ => Instruction::Length {
                    var: self.compile_variable(var),
                    out: self.compile_variable(out),
                },
            },
            cube::Metadata::BufferLength { var } => match var.kind {
                cube::VariableKind::GlobalInputArray(id) => {
                    let offset = self.metadata.buffer_len_index(id);
                    Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                cube::VariableKind::GlobalOutputArray(id) => {
                    let id = self.num_inputs as u32 + id;
                    let offset = self.metadata.buffer_len_index(id);
                    Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                _ => Instruction::Length {
                    var: self.compile_variable(var),
                    out: self.compile_variable(out),
                },
            },
        }
    }

    fn compile_arithmetic(
        &mut self,
        value: cube::Arithmetic,
        out: Option<cube::Variable>,
        instructions: &mut Vec<Instruction>,
        scope: &mut Scope,
    ) {
        let out = out.unwrap();
        match value {
            cube::Arithmetic::Max(op) => instructions.push(Instruction::Max {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Min(op) => instructions.push(Instruction::Min {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Add(op) => instructions.push(Instruction::Add {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Fma(op) => instructions.push(Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Modulo(op) => instructions.push(Instruction::Modulo {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Sub(op) => instructions.push(Instruction::Sub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Mul(op) => instructions.push(Instruction::Mul {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Div(op) => instructions.push(Instruction::Div {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Abs(op) => instructions.push(Instruction::Abs {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Exp(op) => instructions.push(Instruction::Exp {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Log(op) => instructions.push(Instruction::Log {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Log1p(op) => instructions.push(Instruction::Log1p {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Cos(op) => instructions.push(Instruction::Cos {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Sin(op) => instructions.push(Instruction::Sin {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Tanh(op) => instructions.push(Instruction::Tanh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Powf(op) => instructions.push(Instruction::Powf {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Sqrt(op) => instructions.push(Instruction::Sqrt {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Round(op) => instructions.push(Instruction::Round {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Floor(op) => instructions.push(Instruction::Floor {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Ceil(op) => instructions.push(Instruction::Ceil {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Erf(op) => {
                let mut scope = scope.child();
                expand_erf(&mut scope, op.input, out);
                instructions.extend(self.compile_scope(&mut scope));
            }
            cube::Arithmetic::Recip(op) => instructions.push(Instruction::Recip {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Clamp(op) => instructions.push(Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Remainder(op) => instructions.push(Instruction::Remainder {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Neg(op) => instructions.push(Instruction::Negate {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Magnitude(op) => instructions.push(Instruction::Magnitude {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Normalize(op) => instructions.push(Instruction::Normalize {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Dot(op) => instructions.push(Instruction::Dot {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
        }
    }

    fn compile_cmp(
        &mut self,
        value: cube::Comparison,
        out: Option<cube::Variable>,
        instructions: &mut Vec<Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Comparison::Equal(op) => instructions.push(Instruction::Equal {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::Lower(op) => instructions.push(Instruction::Lower {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::Greater(op) => instructions.push(Instruction::Greater {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::LowerEqual(op) => instructions.push(Instruction::LowerEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::GreaterEqual(op) => instructions.push(Instruction::GreaterEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::NotEqual(op) => instructions.push(Instruction::NotEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
        }
    }

    fn compile_bitwise(
        &mut self,
        value: cube::Bitwise,
        out: Option<cube::Variable>,
        instructions: &mut Vec<Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Bitwise::BitwiseOr(op) => instructions.push(Instruction::BitwiseOr {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::BitwiseAnd(op) => instructions.push(Instruction::BitwiseAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::BitwiseXor(op) => instructions.push(Instruction::BitwiseXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::CountOnes(op) => instructions.push(Instruction::CountBits {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::ReverseBits(op) => instructions.push(Instruction::ReverseBits {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::ShiftLeft(op) => instructions.push(Instruction::ShiftLeft {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::ShiftRight(op) => instructions.push(Instruction::ShiftRight {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::BitwiseNot(op) => instructions.push(Instruction::BitwiseNot {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::LeadingZeros(op) => instructions.push(Instruction::LeadingZeros {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::FindFirstSet(op) => instructions.push(Instruction::FindFirstSet {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
        }
    }

    fn compile_operator(
        &mut self,
        value: cube::Operator,
        out: Option<cube::Variable>,
        instructions: &mut Vec<Instruction>,
        scope: &mut cube::Scope,
    ) {
        let out = out.unwrap();
        match value {
            cube::Operator::Select(op) => instructions.push(Instruction::Select {
                cond: self.compile_variable(op.cond),
                then: self.compile_variable(op.then),
                or_else: self.compile_variable(op.or_else),
                out: self.compile_variable(out),
            }),




            cube::Operator::Cast(op) => instructions.push(Instruction::Assign {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Operator::Index(op) => {
                if matches!(self.strategy, ExecutionMode::Checked) && op.lhs.has_length() {
                    let lhs = op.lhs;
                    let rhs = op.rhs;
                    let array_len =
                        *scope.create_local(cube::Item::new(cube::Elem::UInt(cube::UIntKind::U32)));

                    instructions.extend(self.compile_scope(scope));

                    let length = match lhs.has_buffer_length() {
                        true => cube::Metadata::BufferLength { var: lhs },
                        false => cube::Metadata::Length { var: lhs },
                    };
                    instructions.push(self.compile_metadata(length, Some(array_len)));
                    instructions.push(Instruction::CheckedIndex {
                        len: self.compile_variable(array_len),
                        lhs: self.compile_variable(lhs),
                        rhs: self.compile_variable(rhs),
                        out: self.compile_variable(out),
                    });
                } else {
                    instructions.push(Instruction::Index {
                        lhs: self.compile_variable(op.lhs),
                        rhs: self.compile_variable(op.rhs),
                        out: self.compile_variable(out),
                    });
                }
            }
            cube::Operator::UncheckedIndex(op) => instructions.push(Instruction::Index {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Operator::IndexAssign(op) => {
                if let ExecutionMode::Checked = self.strategy {
                    if out.has_length() {
                        expand_checked_index_assign(scope, op.lhs, op.rhs, out);
                        instructions.extend(self.compile_scope(scope));
                        return;
                    }
                };
                instructions.push(Instruction::IndexAssign {
                    lhs: self.compile_variable(op.lhs),
                    rhs: self.compile_variable(op.rhs),
                    out: self.compile_variable(out),
                })
            }
            cube::Operator::UncheckedIndexAssign(op) => {
                instructions.push(Instruction::IndexAssign {
                    lhs: self.compile_variable(op.lhs),
                    rhs: self.compile_variable(op.rhs),
                    out: self.compile_variable(out),
                })
            }
            cube::Operator::And(op) => instructions.push(Instruction::And {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Operator::Or(op) => instructions.push(Instruction::Or {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Operator::Not(op) => instructions.push(Instruction::Not {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Operator::Slice(op) => {
                if matches!(self.strategy, ExecutionMode::Checked) && op.input.has_length() {
                    let input = op.input;
                    let input_len = *scope
                        .create_local_mut(cube::Item::new(cube::Elem::UInt(cube::UIntKind::U32)));
                    instructions.extend(self.compile_scope(scope));

                    let length = match input.has_buffer_length() {
                        true => cube::Metadata::BufferLength { var: input },
                        false => cube::Metadata::Length { var: input },
                    };

                    instructions.push(self.compile_metadata(length, Some(input_len)));
                    instructions.push(Instruction::CheckedSlice {
                        input: self.compile_variable(input),
                        start: self.compile_variable(op.start),
                        end: self.compile_variable(op.end),
                        out: self.compile_variable(out),
                        len: self.compile_variable(input_len),
                    });
                } else {
                    instructions.push(Instruction::Slice {
                        input: self.compile_variable(op.input),
                        start: self.compile_variable(op.start),
                        end: self.compile_variable(op.end),
                        out: self.compile_variable(out),
                    });
                }
            }
            cube::Operator::Bitcast(op) => instructions.push(Instruction::Bitcast {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Operator::InitLine(op) => instructions.push(Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|var| self.compile_variable(var))
                    .collect(),
                out: self.compile_variable(out),
            }),
            cube::Operator::CopyMemory(op) => instructions.push(Instruction::Copy {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
            }),
            cube::Operator::CopyMemoryBulk(op) => instructions.push(Instruction::CopyBulk {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
                len: op.len.as_const().unwrap().as_u32(),
            }),
        }
    }

    fn compile_atomic(
        &mut self,
        atomic: cube::AtomicOp,
        out: Option<cube::Variable>,
    ) -> Instruction {
        let out = out.unwrap();
        match atomic {
            cube::AtomicOp::Add(op) => Instruction::AtomicAdd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Sub(op) => Instruction::AtomicSub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Max(op) => Instruction::AtomicMax {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Min(op) => Instruction::AtomicMin {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::And(op) => Instruction::AtomicAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Or(op) => Instruction::AtomicOr {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Xor(op) => Instruction::AtomicXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Load(op) => Instruction::AtomicLoad {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Store(op) => Instruction::AtomicStore {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Swap(op) => Instruction::AtomicSwap {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::CompareAndSwap(op) => Instruction::AtomicCompareExchangeWeak {
                lhs: self.compile_variable(op.input),
                cmp: self.compile_variable(op.cmp),
                value: self.compile_variable(op.val),
                out: self.compile_variable(out),
            },
        }
    }

    fn compile_location(value: compute::Location, visibility: compute::Visibility) -> AddressSpace {
        match value {
            compute::Location::Storage => {
                match visibility {
                    Visibility::Read => AddressSpace::Constant,
                    Visibility::ReadWrite => AddressSpace::Device,
                }
            },
            compute::Location::Cube => AddressSpace::ThreadGroup,
        }
    }

    fn compile_binding(value: compute::Binding) -> Binding {
        Binding {
            address_space: Self::compile_location(value.location, value.visibility),
            item: Self::compile_item(value.item),
            size: value.size,
        }
    }
}

