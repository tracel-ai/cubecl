use super::Subgroup;
use super::{ConstantArray, shader::ComputeShader};
use super::{Item, LocalArray, SharedMemory};
use crate::compiler::wgsl;

use cubecl_common::ExecutionMode;
use cubecl_core::ir::{ConstantScalarValue, ExpandElement, UIntKind};
use cubecl_core::prelude::{FloatExpand, Line};
use cubecl_core::{
    Metadata, WgpuCompilationOptions, compute,
    ir::{self as cube, Scope},
    prelude::{expand_checked_index_assign, expand_erf},
};
use cubecl_core::{io::read_tensor_checked, prelude::*};

/// Wgsl Compiler.
#[derive(Clone, Default)]
pub struct WgslCompiler {
    metadata: Metadata,
    ext_meta_pos: Vec<u32>,
    local_invocation_index: bool,
    local_invocation_id: bool,
    // TODO: possible cleanup, this bool seems to not be used
    global_invocation_id: bool,
    workgroup_id: bool,
    subgroup_size: bool,
    subgroup_invocation_id: bool,
    id: bool,
    num_workgroups: bool,
    workgroup_id_no_axis: bool,
    workgroup_size_no_axis: bool,
    num_workgroup_no_axis: bool,
    shared_memories: Vec<SharedMemory>,
    const_arrays: Vec<ConstantArray>,
    local_arrays: Vec<LocalArray>,
    #[allow(dead_code)]
    compilation_options: WgpuCompilationOptions,
    strategy: ExecutionMode,
    subgroup_instructions_used: bool,
    f16_used: bool,
}

impl core::fmt::Debug for WgslCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgslCompiler")
    }
}

impl cubecl_core::Compiler for WgslCompiler {
    type Representation = ComputeShader;
    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        shader: compute::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        self.compilation_options = compilation_options.clone();
        self.compile_shader(shader, mode)
    }

    fn elem_size(&self, elem: cube::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "wgsl"
    }
}

impl WgslCompiler {
    fn compile_shader(
        &mut self,
        mut value: compute::KernelDefinition,
        mode: ExecutionMode,
    ) -> wgsl::ComputeShader {
        self.strategy = mode;

        let num_meta = value.buffers.len();

        self.ext_meta_pos = Vec::new();
        let mut num_ext = 0;

        for binding in value.buffers.iter() {
            self.ext_meta_pos.push(num_ext);
            if binding.has_extended_meta {
                num_ext += 1;
            }
        }

        self.metadata = Metadata::new(num_meta as u32, num_ext);

        let instructions = self.compile_scope(&mut value.body);
        let extensions = register_extensions(&instructions);
        let body = wgsl::Body {
            instructions,
            id: self.id,
        };

        wgsl::ComputeShader {
            buffers: value
                .buffers
                .into_iter()
                .map(|it| self.compile_binding(it))
                .collect(),
            scalars: value
                .scalars
                .into_iter()
                .map(|binding| (self.compile_elem(binding.elem), binding.count))
                .collect(),
            shared_memories: self.shared_memories.clone(),
            constant_arrays: self.const_arrays.clone(),
            local_arrays: self.local_arrays.clone(),
            has_metadata: self.metadata.static_len() > 0,
            workgroup_size: value.cube_dim,
            global_invocation_id: self.global_invocation_id || self.id,
            local_invocation_index: self.local_invocation_index,
            local_invocation_id: self.local_invocation_id,
            num_workgroups: self.id
                || self.num_workgroups
                || self.num_workgroup_no_axis
                || self.workgroup_id_no_axis,
            workgroup_id: self.workgroup_id || self.workgroup_id_no_axis,
            subgroup_size: self.subgroup_size,
            subgroup_invocation_id: self.subgroup_invocation_id,
            body,
            extensions,
            num_workgroups_no_axis: self.num_workgroup_no_axis,
            workgroup_id_no_axis: self.workgroup_id_no_axis,
            workgroup_size_no_axis: self.workgroup_size_no_axis,
            subgroup_instructions_used: self.subgroup_instructions_used,
            f16_used: self.f16_used,
            kernel_name: value.options.kernel_name,
        }
    }

    fn compile_item(&mut self, item: cube::Item) -> Item {
        let elem = self.compile_elem(item.elem);
        match item.vectorization.map(|it| it.get()).unwrap_or(1) {
            1 => wgsl::Item::Scalar(elem),
            2 => wgsl::Item::Vec2(elem),
            3 => wgsl::Item::Vec3(elem),
            4 => wgsl::Item::Vec4(elem),
            _ => panic!("Unsupported vectorizations scheme {:?}", item.vectorization),
        }
    }

    fn compile_elem(&mut self, value: cube::Elem) -> wgsl::Elem {
        match value {
            cube::Elem::Float(f) => match f {
                cube::FloatKind::F16 => {
                    self.f16_used = true;
                    wgsl::Elem::F16
                }
                cube::FloatKind::BF16 => panic!("bf16 is not a valid WgpuElement"),
                cube::FloatKind::TF32 => panic!("tf32 is not a valid WgpuElement"),
                cube::FloatKind::Flex32 => wgsl::Elem::F32,
                cube::FloatKind::F32 => wgsl::Elem::F32,
                cube::FloatKind::F64 => wgsl::Elem::F64,
            },
            cube::Elem::Int(i) => match i {
                cube::IntKind::I32 => wgsl::Elem::I32,
                cube::IntKind::I64 => wgsl::Elem::I64,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
            cube::Elem::UInt(kind) => match kind {
                cube::UIntKind::U32 => wgsl::Elem::U32,
                cube::UIntKind::U64 => wgsl::Elem::U64,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
            cube::Elem::Bool => wgsl::Elem::Bool,
            cube::Elem::AtomicFloat(i) => match i {
                cube::FloatKind::F32 => wgsl::Elem::AtomicF32,
                kind => panic!("atomic<{kind:?}> is not a valid WgpuElement"),
            },
            cube::Elem::AtomicInt(i) => match i {
                cube::IntKind::I32 => wgsl::Elem::AtomicI32,
                kind => panic!("atomic<{kind:?}> is not a valid WgpuElement"),
            },
            cube::Elem::AtomicUInt(kind) => match kind {
                cube::UIntKind::U32 => wgsl::Elem::AtomicU32,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
        }
    }

    fn ext_meta_pos(&self, var: &cube::Variable) -> u32 {
        let pos = var.index().expect("Variable should have index");
        self.ext_meta_pos[pos as usize]
    }

    pub(crate) fn compile_variable(&mut self, value: cube::Variable) -> wgsl::Variable {
        let item = value.item;
        match value.kind {
            cube::VariableKind::GlobalInputArray(id) => {
                wgsl::Variable::GlobalInputArray(id, self.compile_item(item))
            }
            cube::VariableKind::GlobalScalar(id) => {
                wgsl::Variable::GlobalScalar(id, self.compile_elem(item.elem), item.elem)
            }
            cube::VariableKind::LocalMut { id } | cube::VariableKind::Versioned { id, .. } => {
                wgsl::Variable::LocalMut {
                    id,
                    item: self.compile_item(item),
                }
            }
            cube::VariableKind::LocalConst { id } => wgsl::Variable::LocalConst {
                id,
                item: self.compile_item(item),
            },
            cube::VariableKind::GlobalOutputArray(id) => {
                wgsl::Variable::GlobalOutputArray(id, self.compile_item(item))
            }
            cube::VariableKind::ConstantScalar(value) => {
                wgsl::Variable::ConstantScalar(value, self.compile_elem(value.elem()))
            }
            cube::VariableKind::SharedMemory {
                id,
                length,
                alignment,
            } => {
                let item = self.compile_item(item);
                if !self.shared_memories.iter().any(|s| s.index == id) {
                    self.shared_memories
                        .push(SharedMemory::new(id, item, length, alignment));
                }
                wgsl::Variable::SharedMemory(id, item, length)
            }
            cube::VariableKind::ConstantArray { id, length } => {
                let item = self.compile_item(item);
                wgsl::Variable::ConstantArray(id, item, length)
            }
            cube::VariableKind::LocalArray { id, length } => {
                let item = self.compile_item(item);
                if !self.local_arrays.iter().any(|s| s.index == id) {
                    self.local_arrays.push(LocalArray::new(id, item, length));
                }
                wgsl::Variable::LocalArray(id, item, length)
            }
            cube::VariableKind::Builtin(builtin) => match builtin {
                cube::Builtin::AbsolutePos => {
                    self.id = true;
                    wgsl::Variable::Id
                }
                cube::Builtin::UnitPos => {
                    self.local_invocation_index = true;
                    wgsl::Variable::LocalInvocationIndex
                }
                cube::Builtin::UnitPosX => {
                    self.local_invocation_id = true;
                    wgsl::Variable::LocalInvocationIdX
                }
                cube::Builtin::UnitPosY => {
                    self.local_invocation_id = true;
                    wgsl::Variable::LocalInvocationIdY
                }
                cube::Builtin::UnitPosZ => {
                    self.local_invocation_id = true;
                    wgsl::Variable::LocalInvocationIdZ
                }
                cube::Builtin::CubePosX => {
                    self.workgroup_id = true;
                    wgsl::Variable::WorkgroupIdX
                }
                cube::Builtin::CubePosY => {
                    self.workgroup_id = true;
                    wgsl::Variable::WorkgroupIdY
                }
                cube::Builtin::CubePosZ => {
                    self.workgroup_id = true;
                    wgsl::Variable::WorkgroupIdZ
                }
                cube::Builtin::CubePosCluster
                | cube::Builtin::CubePosClusterX
                | cube::Builtin::CubePosClusterY
                | cube::Builtin::CubePosClusterZ => self.constant_var(1),
                cube::Builtin::AbsolutePosX => {
                    self.global_invocation_id = true;
                    wgsl::Variable::GlobalInvocationIdX
                }
                cube::Builtin::AbsolutePosY => {
                    self.global_invocation_id = true;
                    wgsl::Variable::GlobalInvocationIdY
                }
                cube::Builtin::AbsolutePosZ => {
                    self.global_invocation_id = true;
                    wgsl::Variable::GlobalInvocationIdZ
                }
                cube::Builtin::CubeDimX => wgsl::Variable::WorkgroupSizeX,
                cube::Builtin::CubeDimY => wgsl::Variable::WorkgroupSizeY,
                cube::Builtin::CubeDimZ => wgsl::Variable::WorkgroupSizeZ,
                cube::Builtin::CubeClusterDim
                | cube::Builtin::CubeClusterDimX
                | cube::Builtin::CubeClusterDimY
                | cube::Builtin::CubeClusterDimZ => self.constant_var(1),
                cube::Builtin::CubeCountX => {
                    self.num_workgroups = true;
                    wgsl::Variable::NumWorkgroupsX
                }
                cube::Builtin::CubeCountY => {
                    self.num_workgroups = true;
                    wgsl::Variable::NumWorkgroupsY
                }
                cube::Builtin::CubeCountZ => {
                    self.num_workgroups = true;
                    wgsl::Variable::NumWorkgroupsZ
                }
                cube::Builtin::CubePos => {
                    self.workgroup_id_no_axis = true;
                    wgsl::Variable::WorkgroupId
                }
                cube::Builtin::CubeDim => {
                    self.workgroup_size_no_axis = true;
                    wgsl::Variable::WorkgroupSize
                }
                cube::Builtin::CubeCount => {
                    self.num_workgroup_no_axis = true;
                    wgsl::Variable::NumWorkgroups
                }
                cube::Builtin::PlaneDim => {
                    self.subgroup_size = true;
                    wgsl::Variable::SubgroupSize
                }
                cube::Builtin::UnitPosPlane => {
                    self.subgroup_invocation_id = true;
                    wgsl::Variable::SubgroupInvocationId
                }
            },
            cube::VariableKind::Matrix { .. } => {
                panic!("Cooperative matrix-multiply and accumulate not supported.")
            }
            cube::VariableKind::Pipeline { .. } => {
                panic!("Pipeline not supported.")
            }
            cube::VariableKind::Barrier { .. } => {
                panic!("Barrier not supported.")
            }
            cube::VariableKind::TensorMap(_) => panic!("Tensor map not supported."),
        }
    }

    fn constant_var(&mut self, value: u32) -> wgsl::Variable {
        let var = cube::Variable::constant(ConstantScalarValue::UInt(value as u64, UIntKind::U32));
        self.compile_variable(var)
    }

    fn compile_scope(&mut self, scope: &mut cube::Scope) -> Vec<wgsl::Instruction> {
        let mut instructions = Vec::new();

        let const_arrays = scope
            .const_arrays
            .drain(..)
            .map(|(var, values)| ConstantArray {
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
            instructions.push(wgsl::Instruction::DeclareVariable {
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
        instructions: &mut Vec<wgsl::Instruction>,
        operation: cube::Operation,
        out: Option<cube::Variable>,
        scope: &mut cube::Scope,
    ) {
        match operation {
            cube::Operation::Copy(variable) => instructions.push(wgsl::Instruction::Assign {
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
            cube::Operation::NonSemantic(_) => {}
            cube::Operation::Barrier(_) => {
                panic!("Barrier isn't supported on wgpu.")
            }
            cube::Operation::Tma(_) => panic!("TMA isn't supported on wgpu."),
        }
    }

    fn compile_subgroup(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
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

        instructions.push(wgsl::Instruction::Subgroup(op));
    }

    fn compile_branch(&mut self, instructions: &mut Vec<wgsl::Instruction>, branch: cube::Branch) {
        match branch {
            cube::Branch::If(mut op) => instructions.push(wgsl::Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            cube::Branch::IfElse(mut op) => instructions.push(wgsl::Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            cube::Branch::Switch(mut op) => instructions.push(wgsl::Instruction::Switch {
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
            cube::Branch::Return => instructions.push(wgsl::Instruction::Return),
            cube::Branch::Break => instructions.push(wgsl::Instruction::Break),
            cube::Branch::RangeLoop(mut range_loop) => {
                instructions.push(wgsl::Instruction::RangeLoop {
                    i: self.compile_variable(range_loop.i),
                    start: self.compile_variable(range_loop.start),
                    end: self.compile_variable(range_loop.end),
                    step: range_loop.step.map(|it| self.compile_variable(it)),
                    inclusive: range_loop.inclusive,
                    instructions: self.compile_scope(&mut range_loop.scope),
                })
            }
            cube::Branch::Loop(mut op) => instructions.push(wgsl::Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }

    fn compile_synchronization(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        synchronization: cube::Synchronization,
    ) {
        match synchronization {
            cube::Synchronization::SyncCube => {
                instructions.push(wgsl::Instruction::WorkgroupBarrier)
            }
            cube::Synchronization::SyncPlane => {
                panic!("Synchronization within a plane is not supported in WGSL")
            }
            cube::Synchronization::SyncStorage => {
                instructions.push(wgsl::Instruction::StorageBarrier)
            }
            cube::Synchronization::SyncProxyShared => panic!("TMA is not supported in WGSL"),
        };
    }

    fn compile_comment(&mut self, instructions: &mut Vec<wgsl::Instruction>, content: String) {
        instructions.push(wgsl::Instruction::Comment { content })
    }

    fn compile_metadata(
        &mut self,
        metadata: cube::Metadata,
        out: Option<cube::Variable>,
    ) -> wgsl::Instruction {
        let out = out.unwrap();
        match metadata {
            cube::Metadata::Rank { var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.metadata.rank_index(position);
                wgsl::Instruction::Metadata {
                    out: self.compile_variable(out),
                    info_offset: self.compile_variable(offset.into()),
                }
            }
            cube::Metadata::Stride { dim, var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.metadata.stride_offset_index(position);
                wgsl::Instruction::ExtendedMeta {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            cube::Metadata::Shape { dim, var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.metadata.shape_offset_index(position);
                wgsl::Instruction::ExtendedMeta {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            cube::Metadata::Length { var } => match var.kind {
                cube::VariableKind::GlobalInputArray(id) => {
                    let offset = self.metadata.len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                cube::VariableKind::GlobalOutputArray(id) => {
                    let offset = self.metadata.len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                _ => wgsl::Instruction::Length {
                    var: self.compile_variable(var),
                    out: self.compile_variable(out),
                },
            },
            cube::Metadata::BufferLength { var } => match var.kind {
                cube::VariableKind::GlobalInputArray(id) => {
                    let offset = self.metadata.buffer_len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                cube::VariableKind::GlobalOutputArray(id) => {
                    let offset = self.metadata.buffer_len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                _ => wgsl::Instruction::Length {
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
        instructions: &mut Vec<wgsl::Instruction>,
        scope: &mut Scope,
    ) {
        let out = out.unwrap();
        match value {
            cube::Arithmetic::Max(op) => instructions.push(wgsl::Instruction::Max {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Min(op) => instructions.push(wgsl::Instruction::Min {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Add(op) => instructions.push(wgsl::Instruction::Add {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Fma(op) => instructions.push(wgsl::Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Modulo(op) => instructions.push(wgsl::Instruction::Modulo {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Sub(op) => instructions.push(wgsl::Instruction::Sub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Mul(op) => instructions.push(wgsl::Instruction::Mul {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Div(op) => instructions.push(wgsl::Instruction::Div {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Abs(op) => instructions.push(wgsl::Instruction::Abs {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Exp(op) => instructions.push(wgsl::Instruction::Exp {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Log(op) => instructions.push(wgsl::Instruction::Log {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Log1p(op) => instructions.push(wgsl::Instruction::Log1p {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Cos(op) => instructions.push(wgsl::Instruction::Cos {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Sin(op) => instructions.push(wgsl::Instruction::Sin {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Tanh(op) => instructions.push(wgsl::Instruction::Tanh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Powf(op) => instructions.push(wgsl::Instruction::Powf {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Sqrt(op) => instructions.push(wgsl::Instruction::Sqrt {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Round(op) => instructions.push(wgsl::Instruction::Round {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Floor(op) => instructions.push(wgsl::Instruction::Floor {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Ceil(op) => instructions.push(wgsl::Instruction::Ceil {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Erf(op) => {
                let mut scope = scope.child();
                expand_erf(&mut scope, op.input, out);
                instructions.extend(self.compile_scope(&mut scope));
            }
            cube::Arithmetic::MulHi(op) => {
                let mut scope = scope.child();
                match self.compilation_options.supports_u64 {
                    true => expand_himul_64(&mut scope, op.lhs, op.rhs, out),
                    false => expand_himul_sim(&mut scope, op.lhs, op.rhs, out),
                }
                instructions.extend(self.compile_scope(&mut scope));
            }
            cube::Arithmetic::Recip(op) => instructions.push(wgsl::Instruction::Recip {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Clamp(op) => instructions.push(wgsl::Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Remainder(op) => instructions.push(wgsl::Instruction::Remainder {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Neg(op) => instructions.push(wgsl::Instruction::Negate {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Magnitude(op) => instructions.push(wgsl::Instruction::Magnitude {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Normalize(op) => instructions.push(wgsl::Instruction::Normalize {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Arithmetic::Dot(op) => instructions.push(wgsl::Instruction::Dot {
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
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Comparison::Equal(op) => instructions.push(wgsl::Instruction::Equal {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::Lower(op) => instructions.push(wgsl::Instruction::Lower {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::Greater(op) => instructions.push(wgsl::Instruction::Greater {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::LowerEqual(op) => instructions.push(wgsl::Instruction::LowerEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Comparison::GreaterEqual(op) => {
                instructions.push(wgsl::Instruction::GreaterEqual {
                    lhs: self.compile_variable(op.lhs),
                    rhs: self.compile_variable(op.rhs),
                    out: self.compile_variable(out),
                })
            }
            cube::Comparison::NotEqual(op) => instructions.push(wgsl::Instruction::NotEqual {
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
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Bitwise::BitwiseOr(op) => instructions.push(wgsl::Instruction::BitwiseOr {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::BitwiseAnd(op) => instructions.push(wgsl::Instruction::BitwiseAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::BitwiseXor(op) => instructions.push(wgsl::Instruction::BitwiseXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::CountOnes(op) => instructions.push(wgsl::Instruction::CountBits {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::ReverseBits(op) => instructions.push(wgsl::Instruction::ReverseBits {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::ShiftLeft(op) => instructions.push(wgsl::Instruction::ShiftLeft {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::ShiftRight(op) => instructions.push(wgsl::Instruction::ShiftRight {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::BitwiseNot(op) => instructions.push(wgsl::Instruction::BitwiseNot {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::LeadingZeros(op) => instructions.push(wgsl::Instruction::LeadingZeros {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Bitwise::FindFirstSet(op) => instructions.push(wgsl::Instruction::FindFirstSet {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
        }
    }

    fn compile_operator(
        &mut self,
        value: cube::Operator,
        out: Option<cube::Variable>,
        instructions: &mut Vec<wgsl::Instruction>,
        scope: &mut cube::Scope,
    ) {
        let out = out.unwrap();
        match value {
            cube::Operator::Cast(op) => instructions.push(wgsl::Instruction::Assign {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Operator::Index(op) => {
                if matches!(self.strategy, ExecutionMode::Checked)
                    && op.list.has_length()
                    && !out.elem().is_atomic()
                {
                    let list = ExpandElement::Plain(op.list);
                    let index = ExpandElement::Plain(op.index);
                    scope.register_elem::<FloatExpand<0>>(op.list.elem());

                    let mut child_scope = scope.child();
                    let input = read_tensor_checked::expand::<Line<FloatExpand<0>>>(
                        &mut child_scope,
                        list.into(),
                        index.into(),
                    );
                    for inst in self.compile_scope(&mut child_scope) {
                        instructions.push(inst);
                    }

                    instructions.push(wgsl::Instruction::Assign {
                        input: self.compile_variable(input.into_variable()),
                        out: self.compile_variable(out),
                    })
                } else {
                    instructions.push(wgsl::Instruction::Index {
                        lhs: self.compile_variable(op.list),
                        rhs: self.compile_variable(op.index),
                        out: self.compile_variable(out),
                    });
                }
            }
            cube::Operator::UncheckedIndex(op) => instructions.push(wgsl::Instruction::Index {
                lhs: self.compile_variable(op.list),
                rhs: self.compile_variable(op.index),
                out: self.compile_variable(out),
            }),
            cube::Operator::IndexAssign(op) => {
                if let ExecutionMode::Checked = self.strategy {
                    if out.has_length() {
                        expand_checked_index_assign(scope, op.index, op.value, out);
                        instructions.extend(self.compile_scope(scope));
                        return;
                    }
                };
                instructions.push(wgsl::Instruction::IndexAssign {
                    index: self.compile_variable(op.index),
                    rhs: self.compile_variable(op.value),
                    out: self.compile_variable(out),
                })
            }
            cube::Operator::UncheckedIndexAssign(op) => {
                instructions.push(wgsl::Instruction::IndexAssign {
                    index: self.compile_variable(op.index),
                    rhs: self.compile_variable(op.value),
                    out: self.compile_variable(out),
                })
            }
            cube::Operator::And(op) => instructions.push(wgsl::Instruction::And {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Operator::Or(op) => instructions.push(wgsl::Instruction::Or {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            cube::Operator::Not(op) => instructions.push(wgsl::Instruction::Not {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Operator::Reinterpret(op) => instructions.push(wgsl::Instruction::Bitcast {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            cube::Operator::InitLine(op) => instructions.push(wgsl::Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|var| self.compile_variable(var))
                    .collect(),
                out: self.compile_variable(out),
            }),
            cube::Operator::CopyMemory(op) => instructions.push(wgsl::Instruction::Copy {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
            }),
            cube::Operator::CopyMemoryBulk(op) => instructions.push(wgsl::Instruction::CopyBulk {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
                len: op.len.as_const().unwrap().as_u32(),
            }),
            cube::Operator::Select(op) => instructions.push(wgsl::Instruction::Select {
                cond: self.compile_variable(op.cond),
                then: self.compile_variable(op.then),
                or_else: self.compile_variable(op.or_else),
                out: self.compile_variable(out),
            }),
        }
    }

    fn compile_atomic(
        &mut self,
        atomic: cube::AtomicOp,
        out: Option<cube::Variable>,
    ) -> wgsl::Instruction {
        let out = out.unwrap();
        match atomic {
            cube::AtomicOp::Add(op) => wgsl::Instruction::AtomicAdd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Sub(op) => wgsl::Instruction::AtomicSub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Max(op) => wgsl::Instruction::AtomicMax {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Min(op) => wgsl::Instruction::AtomicMin {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::And(op) => wgsl::Instruction::AtomicAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Or(op) => wgsl::Instruction::AtomicOr {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Xor(op) => wgsl::Instruction::AtomicXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Load(op) => wgsl::Instruction::AtomicLoad {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Store(op) => wgsl::Instruction::AtomicStore {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::Swap(op) => wgsl::Instruction::AtomicSwap {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            cube::AtomicOp::CompareAndSwap(op) => wgsl::Instruction::AtomicCompareExchangeWeak {
                lhs: self.compile_variable(op.input),
                cmp: self.compile_variable(op.cmp),
                value: self.compile_variable(op.val),
                out: self.compile_variable(out),
            },
        }
    }

    fn compile_location(value: compute::Location) -> wgsl::Location {
        match value {
            compute::Location::Storage => wgsl::Location::Storage,
            compute::Location::Cube => wgsl::Location::Workgroup,
        }
    }

    fn compile_binding(&mut self, value: compute::Binding) -> wgsl::Binding {
        wgsl::Binding {
            id: value.id,
            visibility: value.visibility,
            location: Self::compile_location(value.location),
            item: self.compile_item(value.item),
            size: value.size,
        }
    }
}

fn register_extensions(instructions: &[wgsl::Instruction]) -> Vec<wgsl::Extension> {
    let mut extensions = Vec::new();

    let mut register_extension = |extension: wgsl::Extension| {
        if !extensions.contains(&extension) {
            extensions.push(extension);
        }
    };

    // Since not all instructions are native to WGSL, we need to add the custom ones.
    for instruction in instructions {
        match instruction {
            wgsl::Instruction::Powf { lhs: _, rhs, out } => {
                register_extension(wgsl::Extension::PowfPrimitive(out.item()));

                if rhs.is_always_scalar() || rhs.item().vectorization_factor() == 1 {
                    register_extension(wgsl::Extension::PowfScalar(out.item()));
                } else {
                    register_extension(wgsl::Extension::Powf(out.item()));
                }
            }
            #[cfg(target_os = "macos")]
            wgsl::Instruction::Tanh { input, out: _ } => {
                register_extension(wgsl::Extension::SafeTanh(input.item()))
            }
            wgsl::Instruction::If { instructions, .. } => {
                for extension in register_extensions(instructions) {
                    register_extension(extension);
                }
            }
            wgsl::Instruction::IfElse {
                instructions_if,
                instructions_else,
                ..
            } => {
                for extension in register_extensions(instructions_if) {
                    register_extension(extension);
                }
                for extension in register_extensions(instructions_else) {
                    register_extension(extension);
                }
            }
            wgsl::Instruction::Loop { instructions } => {
                for extension in register_extensions(instructions) {
                    register_extension(extension);
                }
            }
            wgsl::Instruction::RangeLoop { instructions, .. } => {
                for extension in register_extensions(instructions) {
                    register_extension(extension);
                }
            }
            _ => {}
        }
    }

    extensions
}
