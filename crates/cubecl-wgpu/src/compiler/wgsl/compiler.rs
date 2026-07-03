use super::Item;
use super::Subgroup;
use super::shader::ComputeShader;
use crate::compiler::wgsl::{self, SharedValue};

use cubecl_common::backtrace::BackTrace;
use cubecl_core::CompilerProfiler;
use cubecl_core::ir::{Processor, UIntKind};
use cubecl_core::{
    Info,
    post_processing::{
        checked_io::CheckedIoVisitor, optimize_scope, saturating::SaturatingArithmeticProcessor,
        unroll::UnrollVisitor,
    },
};
use cubecl_core::{
    Metadata, WgpuCompilationOptions,
    ir::{self as cube, Scope},
    prelude::expand_erf,
};
use cubecl_core::{post_processing::disaggregate::DisaggregateVisitor, prelude::*};
use cubecl_ir::AddressSpace;
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::kernel;
use hashbrown::HashMap;

pub const MAX_VECTOR_SIZE: usize = 4;

/// Wgsl Compiler.
#[derive(Clone, Default)]
pub struct WgslCompiler {
    kernel_name: String,
    info: Info,
    ext_meta_pos: HashMap<cube::Value, u32>,
    buffer_vis: Vec<Visibility>,
    local_invocation_index: bool,
    local_invocation_id: bool,
    global_invocation_id: bool,
    workgroup_id: bool,
    subgroup_size: bool,
    subgroup_id: bool,
    subgroup_invocation_id: bool,
    id: bool,
    num_workgroups: bool,
    workgroup_id_no_axis: bool,
    workgroup_size_no_axis: bool,
    num_workgroup_no_axis: bool,
    shared_values: Vec<SharedValue>,
    #[allow(dead_code)]
    compilation_options: WgpuCompilationOptions,
    strategy: ExecutionMode,
    subgroup_instructions_used: bool,
    f16_used: bool,
    profiler: CompilerProfiler,
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
        shader: kernel::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        self.compilation_options = *compilation_options;
        self.compile_shader(shader, mode, address_type)
    }

    fn elem_size(&self, elem: cube::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "wgsl"
    }
}

impl WgslCompiler {
    fn compile_shader(
        &mut self,
        value: kernel::KernelDefinition,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<wgsl::ComputeShader, CompilationError> {
        let errors = value.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile wgsl kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        self.strategy = mode;
        self.kernel_name = value.options.kernel_name.clone();

        let num_meta = value.buffers.len();

        self.ext_meta_pos = HashMap::new();
        let mut num_ext = 0;

        for binding in value.buffers.iter() {
            self.ext_meta_pos.insert(binding.value, num_ext);
            if binding.has_extended_meta {
                num_ext += 1;
            }
        }

        let metadata = Metadata::new(num_meta as u32, num_ext);
        self.info = Info::new(&value.scalars, metadata, address_type);

        CheckedIoVisitor::new(self.strategy, self.kernel_name.clone()).apply(&value.body);
        DisaggregateVisitor::apply(&value.body);
        UnrollVisitor::new(MAX_VECTOR_SIZE).apply(&value.body);

        self.buffer_vis = optimize_scope(&value.body).into();
        self.buffer_vis
            .resize(value.num_global_buffers(), Visibility::Read);

        self.setup_profiler(&value.body);
        let mut instructions = self.compile_scope(&value.body);
        self.profile(&value.body, &mut instructions);

        let address_type = self.compile_storage_type(address_type);
        let extensions = register_extensions(&instructions);

        let body = wgsl::Body {
            instructions,
            id: self.id,
            address_type,
        };

        Ok(wgsl::ComputeShader {
            address_type,
            buffers: value
                .buffers
                .into_iter()
                .map(|mut it| {
                    // This is safe when combined with the unroll transform that adjusts all indices.
                    // Must not be used alone
                    if it.value.ty.vector_size() > MAX_VECTOR_SIZE {
                        it.value.ty = it.value.ty.with_vector_size(MAX_VECTOR_SIZE);
                    }
                    self.compile_binding(it)
                })
                .collect(),
            scalars: value
                .scalars
                .into_iter()
                .map(|binding| (self.compile_storage_type(binding.ty), binding.count))
                .collect(),
            shared_values: self.shared_values.clone(),
            static_meta_len: self.info.metadata.static_len() as usize,
            info: self.info.clone(),
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
            subgroup_id: self.subgroup_id,
            subgroup_invocation_id: self.subgroup_invocation_id,
            body,
            extensions,
            num_workgroups_no_axis: self.num_workgroup_no_axis,
            workgroup_id_no_axis: self.workgroup_id_no_axis,
            workgroup_size_no_axis: self.workgroup_size_no_axis,
            subgroup_instructions_used: self.subgroup_instructions_used,
            f16_used: self.f16_used,
            kernel_name: value.options.kernel_name,
        })
    }

    fn compile_type(&mut self, item: cube::Type) -> Item {
        match item {
            cube::Type::Scalar(ty) => wgsl::Item::Scalar(self.compile_storage_type(ty)),
            cube::Type::Vector(ty, size) => {
                let elem = self.compile_storage_type(ty.storage_type());
                wgsl::Item::Vector(elem, size)
            }
            cube::Type::Atomic(ty) => {
                let inner = self.compile_type(*ty);
                wgsl::Item::Atomic(inner.intern())
            }
            cube::Type::Pointer(ty, class) => {
                let inner = self.compile_type(*ty);
                let class = self.compile_pointer_class(class);
                wgsl::Item::Pointer(inner.intern(), class)
            }
            cube::Type::Array(ty, size) => {
                let inner = self.compile_type(*ty);
                wgsl::Item::Array(inner.intern(), size)
            }
            cube::Type::DynamicArray(ty) => {
                let inner = self.compile_type(*ty);
                wgsl::Item::DynamicArray(inner.intern())
            }
            cube::Type::Opaque(_) => unimplemented!("Can't compile opaque type"),
            cube::Type::Semantic(_) => unimplemented!("Can't compile semantic type"),
            cube::Type::Matrix(_) => unimplemented!("Matrices not yet supported in WGSL"),
            cube::Type::Aggregate(_) => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    fn compile_storage_type(&mut self, ty: cube::StorageType) -> wgsl::Elem {
        match ty {
            cube::StorageType::Scalar(ty) => self.compile_elem(ty),
            cube::StorageType::Packed(_, _) => {
                unimplemented!("Packed types not yet supported in WGSL")
            }
        }
    }

    fn compile_elem(&mut self, value: cube::ElemType) -> wgsl::Elem {
        match value {
            cube::ElemType::Float(f) => match f {
                cube::FloatKind::E2M1
                | cube::FloatKind::E2M3
                | cube::FloatKind::E3M2
                | cube::FloatKind::E4M3
                | cube::FloatKind::E5M2
                | cube::FloatKind::UE8M0 => panic!("Minifloat is not a valid WgpuElement"),
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
            cube::ElemType::Int(i) => match i {
                cube::IntKind::I32 => wgsl::Elem::I32,
                cube::IntKind::I64 => wgsl::Elem::I64,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
            cube::ElemType::UInt(kind) => match kind {
                cube::UIntKind::U32 => wgsl::Elem::U32,
                cube::UIntKind::U64 => wgsl::Elem::U64,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
            cube::ElemType::Bool => wgsl::Elem::Bool,
        }
    }

    fn compile_pointer_class(&self, class: cube::AddressSpace) -> wgsl::PointerClass {
        match class {
            cubecl_ir::AddressSpace::Global(id) => {
                wgsl::PointerClass::Global(self.buffer_vis[id as usize])
            }
            cubecl_ir::AddressSpace::Shared => wgsl::PointerClass::Shared,
            cubecl_ir::AddressSpace::Local => wgsl::PointerClass::Local,
        }
    }

    fn ext_meta_pos(&self, val: &cube::Value) -> u32 {
        self.ext_meta_pos[val]
    }

    pub(crate) fn compile_value(&mut self, value: cube::Value) -> wgsl::Value {
        let item = value.ty;
        match value.kind {
            cube::ValueKind::Value { id } => wgsl::Value::Value {
                id,
                item: self.compile_type(item),
            },
            cube::ValueKind::Constant(value) => {
                wgsl::Value::Constant(value, self.compile_type(item))
            }
        }
    }

    fn constant_var(&mut self, value: u32) -> wgsl::Value {
        let val = cube::Value::constant(value.into(), UIntKind::U32);
        self.compile_value(val)
    }

    fn setup_profiler(&mut self, scope: &cube::Scope) {
        if !scope.profile.enabled {
            return;
        }

        let counter = scope
            .profile
            .counters_buffer
            .expect("Profiling counters buffer should be initialized");

        self.profiler.set_counter(counter);
    }

    fn profile(&mut self, scope: &cube::Scope, instructions: &mut Vec<wgsl::Instruction>) {
        if !scope.profile.enabled {
            return;
        }

        let (declare_instructions, flush_instructions) =
            self.profiler.profile(&scope.state().allocator);

        let mut declare_wgsl_instructions = Vec::new();

        for instruction in declare_instructions {
            self.compile_operation(
                &mut declare_wgsl_instructions,
                instruction.operation,
                instruction.out,
                scope,
            );
        }

        let mut old_instructions = core::mem::replace(instructions, declare_wgsl_instructions);
        instructions.append(&mut old_instructions);

        for instruction in flush_instructions {
            self.compile_operation(instructions, instruction.operation, instruction.out, scope);
        }
    }

    fn compile_scope(&mut self, scope: &cube::Scope) -> Vec<wgsl::Instruction> {
        let mut instructions = Vec::new();

        let saturating: Box<dyn Processor> = Box::new(SaturatingArithmeticProcessor::new(true));
        let processing = scope.process([&*saturating]);

        processing
            .instructions
            .into_iter()
            .for_each(|op| self.process_operation(&mut instructions, op.operation, op.out, scope));

        instructions
    }

    fn process_operation(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        operation: cube::Operation,
        out: Option<cube::Value>,
        scope: &cube::Scope,
    ) {
        if scope.profile.enabled {
            let new_instructions =
                self.profiler
                    .profile_operation(&operation, out.as_ref(), &scope.state().allocator);

            for instruction in new_instructions {
                self.compile_operation(instructions, instruction.operation, instruction.out, scope);
            }
        }

        self.compile_operation(instructions, operation, out, scope);
    }

    fn compile_operation(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        operation: cube::Operation,
        out: Option<cube::Value>,
        scope: &cube::Scope,
    ) {
        match operation {
            cube::Operation::Copy(value) => instructions.push(wgsl::Instruction::Assign {
                input: self.compile_value(value),
                out: self.compile_value(out.unwrap()),
            }),
            cube::Operation::DeclareVariable {
                value_ty,
                addr_space: AddressSpace::Local,
                ..
            } => instructions.push(wgsl::Instruction::DeclareVariable {
                val: self.compile_value(out.unwrap()),
                value_ty: self.compile_type(value_ty),
            }),
            cube::Operation::DeclareVariable {
                value_ty,
                addr_space: AddressSpace::Shared,
                alignment,
            } => {
                let ty = self.compile_type(value_ty);
                let value = self.compile_value(out.unwrap());
                self.shared_values
                    .push(SharedValue::new(ty, value, alignment as u32));
            }
            cube::Operation::DeclareVariable { addr_space, .. } => {
                unimplemented!("Unsupported declare address space {addr_space}")
            }
            cube::Operation::Memory(memory) => self.compile_memory(memory, out, instructions),
            cube::Operation::Arithmetic(op) => {
                self.compile_arithmetic(op, out, instructions, scope)
            }
            cube::Operation::Comparison(op) => self.compile_cmp(op, out, instructions),
            cube::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
            cube::Operation::Operator(op) => self.compile_operator(op, out, instructions),
            cube::Operation::Atomic(op) => instructions.push(self.compile_atomic(op, out)),
            cube::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            cube::Operation::Branch(val) => self.compile_branch(instructions, val),
            cube::Operation::Synchronization(val) => {
                self.compile_synchronization(instructions, val)
            }
            cube::Operation::WorkgroupUniformLoad(op) => {
                instructions.push(wgsl::Instruction::WorkgroupUniformLoad {
                    input: self.compile_value(op),
                    out: self.compile_value(out.unwrap()),
                });
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
            cube::Operation::TensorIndexing(_) => panic!("TMA isn't supported on wgpu."),
            cube::Operation::Marker(_) => {}
            cube::Operation::ConstructAggregate(..)
            | cube::Operation::ExtractAggregateField(..) => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    fn compile_subgroup(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        subgroup: cube::Plane,
        out: Option<cube::Value>,
    ) {
        self.subgroup_instructions_used = true;

        let out = out.unwrap();
        let op = match subgroup {
            cube::Plane::Elect => Subgroup::Elect {
                out: self.compile_value(out),
            },
            cube::Plane::All(op) => Subgroup::All {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::Any(op) => Subgroup::Any {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::Ballot(op) => Subgroup::Ballot {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },

            cube::Plane::Broadcast(op) => Subgroup::Broadcast {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            },

            cube::Plane::Sum(op) => Subgroup::Sum {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },

            cube::Plane::ExclusiveSum(op) => Subgroup::ExclusiveSum {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::InclusiveSum(op) => Subgroup::InclusiveSum {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::Prod(op) => Subgroup::Prod {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::ExclusiveProd(op) => Subgroup::ExclusiveProd {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::InclusiveProd(op) => Subgroup::InclusiveProd {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::Min(op) => Subgroup::Min {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::Max(op) => Subgroup::Max {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            },
            cube::Plane::Shuffle(op) => Subgroup::Shuffle {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            },
            cube::Plane::ShuffleXor(op) => Subgroup::ShuffleXor {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            },
            cube::Plane::ShuffleUp(op) => Subgroup::ShuffleUp {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            },
            cube::Plane::ShuffleDown(op) => Subgroup::ShuffleDown {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            },
        };

        instructions.push(wgsl::Instruction::Subgroup(op));
    }

    fn compile_branch(&mut self, instructions: &mut Vec<wgsl::Instruction>, branch: cube::Branch) {
        match branch {
            cube::Branch::If(op) => instructions.push(wgsl::Instruction::If {
                cond: self.compile_value(op.cond),
                instructions: self.compile_scope(&op.scope),
            }),
            cube::Branch::IfElse(op) => instructions.push(wgsl::Instruction::IfElse {
                cond: self.compile_value(op.cond),
                instructions_if: self.compile_scope(&op.scope_if),
                instructions_else: self.compile_scope(&op.scope_else),
            }),
            cube::Branch::Switch(op) => instructions.push(wgsl::Instruction::Switch {
                value: self.compile_value(op.value),
                instructions_default: self.compile_scope(&op.scope_default),
                cases: op
                    .cases
                    .into_iter()
                    .map(|(val, scope)| (self.compile_value(val), self.compile_scope(&scope)))
                    .collect(),
            }),
            cube::Branch::Return => instructions.push(wgsl::Instruction::Return),
            // No unreachable hint in WGSL
            cube::Branch::Unreachable => instructions.push(wgsl::Instruction::Return),
            cube::Branch::Break => instructions.push(wgsl::Instruction::Break),
            cube::Branch::RangeLoop(range_loop) => {
                instructions.push(wgsl::Instruction::RangeLoop {
                    i: self.compile_value(range_loop.i),
                    start: self.compile_value(range_loop.start),
                    end: self.compile_value(range_loop.end),
                    step: range_loop.step.map(|it| self.compile_value(it)),
                    inclusive: range_loop.inclusive,
                    instructions: self.compile_scope(&range_loop.scope),
                })
            }
            cube::Branch::Loop(op) => instructions.push(wgsl::Instruction::Loop {
                instructions: self.compile_scope(&op.scope),
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
            cube::Synchronization::SyncAsyncProxyShared => panic!("TMA is not supported in WGSL"),
        };
    }

    fn compile_comment(&mut self, instructions: &mut Vec<wgsl::Instruction>, content: String) {
        instructions.push(wgsl::Instruction::Comment { content })
    }

    fn compile_metadata(
        &mut self,
        metadata: cube::Metadata,
        out: Option<cube::Value>,
    ) -> wgsl::Instruction {
        let out = out.unwrap();
        match metadata {
            cube::Metadata::Stride { dim, list } => {
                let position = self.ext_meta_pos(&list);
                let offset = self.info.metadata.stride_offset_index(position);
                wgsl::Instruction::ExtendedMeta {
                    info_offset: self.compile_value(offset.into()),
                    dim: self.compile_value(dim),
                    out: self.compile_value(out),
                }
            }
            cube::Metadata::Shape { dim, list } => {
                let position = self.ext_meta_pos(&list);
                let offset = self.info.metadata.shape_offset_index(position);
                wgsl::Instruction::ExtendedMeta {
                    info_offset: self.compile_value(offset.into()),
                    dim: self.compile_value(dim),
                    out: self.compile_value(out),
                }
            }
            cube::Metadata::BufferLength { list } => match list.address_space() {
                cube::AddressSpace::Global(id) => {
                    let offset = self.info.metadata.buffer_len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_value(out),
                        info_offset: self.compile_value(offset.into()),
                    }
                }
                _ => wgsl::Instruction::Length {
                    list: self.compile_value(list),
                    out: self.compile_value(out),
                },
            },
        }
    }

    fn compile_memory(
        &mut self,
        value: cube::Memory,
        out: Option<cube::Value>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        match value {
            cube::Memory::Index(op) => {
                instructions.push(wgsl::Instruction::Index {
                    lhs: self.compile_value(op.list),
                    rhs: self.compile_value(op.index),
                    out: self.compile_value(out.unwrap()),
                });
            }
            cube::Memory::Load(value) => instructions.push(wgsl::Instruction::Load {
                input: self.compile_value(value),
                out: self.compile_value(out.unwrap()),
            }),
            cube::Memory::Store(op) => instructions.push(wgsl::Instruction::Store {
                input: self.compile_value(op.value),
                out: self.compile_value(op.ptr),
            }),
            cube::Memory::CopyMemory(op) => instructions.push(wgsl::Instruction::CopyBulk {
                source: self.compile_value(op.source),
                target: self.compile_value(op.target),
                len: op.len as u32,
            }),
        }
    }

    fn compile_arithmetic(
        &mut self,
        value: cube::Arithmetic,
        out: Option<cube::Value>,
        instructions: &mut Vec<wgsl::Instruction>,
        scope: &Scope,
    ) {
        let out = out.unwrap();
        match value {
            cube::Arithmetic::Max(op) => instructions.push(wgsl::Instruction::Max {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Min(op) => instructions.push(wgsl::Instruction::Min {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Add(op) => instructions.push(wgsl::Instruction::Add {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::SaturatingAdd(_) => {
                unreachable!("Saturating add should be removed by processor");
            }
            cube::Arithmetic::Fma(op) => instructions.push(wgsl::Instruction::Fma {
                a: self.compile_value(op.a),
                b: self.compile_value(op.b),
                c: self.compile_value(op.c),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ModFloor(op) => instructions.push(wgsl::Instruction::ModFloor {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Sub(op) => instructions.push(wgsl::Instruction::Sub {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::SaturatingSub(_) => {
                unreachable!("Saturating sub should be removed by processor");
            }
            cube::Arithmetic::Mul(op) => instructions.push(wgsl::Instruction::Mul {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Div(op) => instructions.push(wgsl::Instruction::Div {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Abs(op) => instructions.push(wgsl::Instruction::Abs {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Exp(op) => instructions.push(wgsl::Instruction::Exp {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Log(op) => instructions.push(wgsl::Instruction::Log {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Log1p(op) => instructions.push(wgsl::Instruction::Log1p {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Expm1(op) => instructions.push(wgsl::Instruction::Expm1 {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Cos(op) => instructions.push(wgsl::Instruction::Cos {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Sin(op) => instructions.push(wgsl::Instruction::Sin {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Tan(op) => instructions.push(wgsl::Instruction::Tan {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Tanh(op) => instructions.push(wgsl::Instruction::Tanh {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Sinh(op) => instructions.push(wgsl::Instruction::Sinh {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Cosh(op) => instructions.push(wgsl::Instruction::Cosh {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcCos(op) => instructions.push(wgsl::Instruction::ArcCos {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcSin(op) => instructions.push(wgsl::Instruction::ArcSin {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcTan(op) => instructions.push(wgsl::Instruction::ArcTan {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcSinh(op) => instructions.push(wgsl::Instruction::ArcSinh {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcCosh(op) => instructions.push(wgsl::Instruction::ArcCosh {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcTanh(op) => instructions.push(wgsl::Instruction::ArcTanh {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Degrees(op) => instructions.push(wgsl::Instruction::Degrees {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Radians(op) => instructions.push(wgsl::Instruction::Radians {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::ArcTan2(op) => instructions.push(wgsl::Instruction::ArcTan2 {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            // No powi in WGSL
            cube::Arithmetic::Powf(op) | cube::Arithmetic::Powi(op) => {
                instructions.push(wgsl::Instruction::Powf {
                    lhs: self.compile_value(op.lhs),
                    rhs: self.compile_value(op.rhs),
                    out: self.compile_value(out),
                })
            }
            cube::Arithmetic::Hypot(op) => {
                let scope = scope.child();
                expand_hypot(&scope, op.lhs, op.rhs, out);
                instructions.extend(self.compile_scope(&scope));
            }
            cube::Arithmetic::Rhypot(op) => {
                let scope = scope.child();
                expand_rhypot(&scope, op.lhs, op.rhs, out);
                instructions.extend(self.compile_scope(&scope));
            }

            cube::Arithmetic::Sqrt(op) => instructions.push(wgsl::Instruction::Sqrt {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::InverseSqrt(op) => {
                instructions.push(wgsl::Instruction::InverseSqrt {
                    input: self.compile_value(op.input),
                    out: self.compile_value(out),
                })
            }
            cube::Arithmetic::Round(op) => instructions.push(wgsl::Instruction::Round {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Floor(op) => instructions.push(wgsl::Instruction::Floor {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Ceil(op) => instructions.push(wgsl::Instruction::Ceil {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Trunc(op) => instructions.push(wgsl::Instruction::Trunc {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Erf(op) => {
                let scope = scope.child();
                expand_erf(&scope, op.input, out);
                instructions.extend(self.compile_scope(&scope));
            }
            cube::Arithmetic::MulHi(op) => {
                let scope = scope.child();
                match self.compilation_options.supports_u64 {
                    true => expand_himul_64(&scope, op.lhs, op.rhs, out),
                    false => expand_himul_sim(&scope, op.lhs, op.rhs, out),
                }
                instructions.extend(self.compile_scope(&scope));
            }
            cube::Arithmetic::Recip(op) => instructions.push(wgsl::Instruction::Recip {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Clamp(op) => instructions.push(wgsl::Instruction::Clamp {
                input: self.compile_value(op.input),
                min_value: self.compile_value(op.min_value),
                max_value: self.compile_value(op.max_value),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Rem(op) => instructions.push(wgsl::Instruction::Remainder {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Neg(op) => instructions.push(wgsl::Instruction::Negate {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Magnitude(op) => instructions.push(wgsl::Instruction::Magnitude {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Normalize(op) => instructions.push(wgsl::Instruction::Normalize {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::Dot(op) => instructions.push(wgsl::Instruction::Dot {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Arithmetic::VectorSum(op) => instructions.push(wgsl::Instruction::VectorSum {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
        }
    }

    fn compile_cmp(
        &mut self,
        value: cube::Comparison,
        out: Option<cube::Value>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Comparison::Equal(op) => instructions.push(wgsl::Instruction::Equal {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Comparison::Lower(op) => instructions.push(wgsl::Instruction::Lower {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Comparison::Greater(op) => instructions.push(wgsl::Instruction::Greater {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Comparison::LowerEqual(op) => instructions.push(wgsl::Instruction::LowerEqual {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Comparison::GreaterEqual(op) => {
                instructions.push(wgsl::Instruction::GreaterEqual {
                    lhs: self.compile_value(op.lhs),
                    rhs: self.compile_value(op.rhs),
                    out: self.compile_value(out),
                })
            }
            cube::Comparison::NotEqual(op) => instructions.push(wgsl::Instruction::NotEqual {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Comparison::IsNan(op) => instructions.push(wgsl::Instruction::IsNan {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Comparison::IsInf(op) => instructions.push(wgsl::Instruction::IsInf {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
        }
    }

    fn compile_bitwise(
        &mut self,
        value: cube::Bitwise,
        out: Option<cube::Value>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Bitwise::BitwiseOr(op) => instructions.push(wgsl::Instruction::BitwiseOr {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Bitwise::BitwiseAnd(op) => instructions.push(wgsl::Instruction::BitwiseAnd {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Bitwise::BitwiseXor(op) => instructions.push(wgsl::Instruction::BitwiseXor {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Bitwise::CountOnes(op) => instructions.push(wgsl::Instruction::CountBits {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Bitwise::ReverseBits(op) => instructions.push(wgsl::Instruction::ReverseBits {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Bitwise::ShiftLeft(op) => instructions.push(wgsl::Instruction::ShiftLeft {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Bitwise::ShiftRight(op) => instructions.push(wgsl::Instruction::ShiftRight {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Bitwise::BitwiseNot(op) => instructions.push(wgsl::Instruction::BitwiseNot {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Bitwise::LeadingZeros(op) => instructions.push(wgsl::Instruction::LeadingZeros {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Bitwise::TrailingZeros(op) => {
                instructions.push(wgsl::Instruction::TrailingZeros {
                    input: self.compile_value(op.input),
                    out: self.compile_value(out),
                })
            }
            cube::Bitwise::FindFirstSet(op) => instructions.push(wgsl::Instruction::FindFirstSet {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
        }
    }

    fn compile_operator(
        &mut self,
        value: cube::Operator,
        out: Option<cube::Value>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            cube::Operator::Cast(op) => instructions.push(wgsl::Instruction::Assign {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),

            cube::Operator::And(op) => instructions.push(wgsl::Instruction::And {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Operator::Or(op) => instructions.push(wgsl::Instruction::Or {
                lhs: self.compile_value(op.lhs),
                rhs: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Operator::Not(op) => instructions.push(wgsl::Instruction::Not {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Operator::Reinterpret(op) => instructions.push(wgsl::Instruction::Bitcast {
                input: self.compile_value(op.input),
                out: self.compile_value(out),
            }),
            cube::Operator::InitVector(op) => instructions.push(wgsl::Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|val| self.compile_value(val))
                    .collect(),
                out: self.compile_value(out),
            }),
            cube::Operator::ExtractComponent(op) => instructions.push(wgsl::Instruction::Extract {
                vector: self.compile_value(op.lhs),
                index: self.compile_value(op.rhs),
                out: self.compile_value(out),
            }),
            cube::Operator::InsertComponent(op) => instructions.push(wgsl::Instruction::Insert {
                vector: self.compile_value(op.vector),
                index: self.compile_value(op.index),
                value: self.compile_value(op.value),
                out: self.compile_value(out),
            }),
            cube::Operator::Select(op) => instructions.push(wgsl::Instruction::Select {
                cond: self.compile_value(op.cond),
                then: self.compile_value(op.then),
                or_else: self.compile_value(op.or_else),
                out: self.compile_value(out),
            }),
            cube::Operator::ReadBuiltin(builtin) => {
                let out = self.compile_value(out);
                let constant = {
                    let out = out.clone();
                    |value| {
                        instructions.push(wgsl::Instruction::Assign { input: value, out });
                    }
                };
                let builtin = match builtin {
                    cube::Builtin::AbsolutePos => {
                        self.id = true;
                        wgsl::Builtin::Id
                    }
                    cube::Builtin::UnitPos => {
                        self.local_invocation_index = true;
                        wgsl::Builtin::LocalInvocationIndex
                    }
                    cube::Builtin::UnitPosX => {
                        self.local_invocation_id = true;
                        wgsl::Builtin::LocalInvocationIdX
                    }
                    cube::Builtin::UnitPosY => {
                        self.local_invocation_id = true;
                        wgsl::Builtin::LocalInvocationIdY
                    }
                    cube::Builtin::UnitPosZ => {
                        self.local_invocation_id = true;
                        wgsl::Builtin::LocalInvocationIdZ
                    }
                    cube::Builtin::CubePosX => {
                        self.workgroup_id = true;
                        wgsl::Builtin::WorkgroupIdX
                    }
                    cube::Builtin::CubePosY => {
                        self.workgroup_id = true;
                        wgsl::Builtin::WorkgroupIdY
                    }
                    cube::Builtin::CubePosZ => {
                        self.workgroup_id = true;
                        wgsl::Builtin::WorkgroupIdZ
                    }
                    cube::Builtin::CubePosCluster
                    | cube::Builtin::CubePosClusterX
                    | cube::Builtin::CubePosClusterY
                    | cube::Builtin::CubePosClusterZ => {
                        constant(self.constant_var(1));
                        return;
                    }
                    cube::Builtin::AbsolutePosX => {
                        self.global_invocation_id = true;
                        wgsl::Builtin::GlobalInvocationIdX
                    }
                    cube::Builtin::AbsolutePosY => {
                        self.global_invocation_id = true;
                        wgsl::Builtin::GlobalInvocationIdY
                    }
                    cube::Builtin::AbsolutePosZ => {
                        self.global_invocation_id = true;
                        wgsl::Builtin::GlobalInvocationIdZ
                    }
                    cube::Builtin::CubeDimX => wgsl::Builtin::WorkgroupSizeX,
                    cube::Builtin::CubeDimY => wgsl::Builtin::WorkgroupSizeY,
                    cube::Builtin::CubeDimZ => wgsl::Builtin::WorkgroupSizeZ,
                    cube::Builtin::CubeClusterDim
                    | cube::Builtin::CubeClusterDimX
                    | cube::Builtin::CubeClusterDimY
                    | cube::Builtin::CubeClusterDimZ => {
                        constant(self.constant_var(1));
                        return;
                    }
                    cube::Builtin::CubeCountX => {
                        self.num_workgroups = true;
                        wgsl::Builtin::NumWorkgroupsX
                    }
                    cube::Builtin::CubeCountY => {
                        self.num_workgroups = true;
                        wgsl::Builtin::NumWorkgroupsY
                    }
                    cube::Builtin::CubeCountZ => {
                        self.num_workgroups = true;
                        wgsl::Builtin::NumWorkgroupsZ
                    }
                    cube::Builtin::CubePos => {
                        self.workgroup_id_no_axis = true;
                        wgsl::Builtin::WorkgroupId
                    }
                    cube::Builtin::CubeDim => {
                        self.workgroup_size_no_axis = true;
                        wgsl::Builtin::WorkgroupSize
                    }
                    cube::Builtin::CubeCount => {
                        self.num_workgroup_no_axis = true;
                        wgsl::Builtin::NumWorkgroups
                    }
                    cube::Builtin::PlaneDim => {
                        self.subgroup_size = true;
                        wgsl::Builtin::SubgroupSize
                    }
                    cube::Builtin::PlanePos => {
                        self.subgroup_id = true;
                        wgsl::Builtin::SubgroupId
                    }
                    cube::Builtin::UnitPosPlane => {
                        self.subgroup_invocation_id = true;
                        wgsl::Builtin::SubgroupInvocationId
                    }
                };
                instructions.push(wgsl::Instruction::ReadBuiltin { builtin, out });
            }
            cube::Operator::ReadScalar(id) => instructions.push(wgsl::Instruction::ReadScalar {
                id,
                out: self.compile_value(out),
            }),
        }
    }

    fn compile_atomic(
        &mut self,
        atomic: cube::AtomicOp,
        out: Option<cube::Value>,
    ) -> wgsl::Instruction {
        match atomic {
            cube::AtomicOp::Add(op) => wgsl::Instruction::AtomicAdd {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Sub(op) => wgsl::Instruction::AtomicSub {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Max(op) => wgsl::Instruction::AtomicMax {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Min(op) => wgsl::Instruction::AtomicMin {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::And(op) => wgsl::Instruction::AtomicAnd {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Or(op) => wgsl::Instruction::AtomicOr {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Xor(op) => wgsl::Instruction::AtomicXor {
                ptr: self.compile_value(op.ptr),
                value: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Load(ptr) => wgsl::Instruction::AtomicLoad {
                input: self.compile_value(ptr),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::Store(op) => wgsl::Instruction::AtomicStore {
                input: self.compile_value(op.value),
                out: self.compile_value(op.ptr),
            },
            cube::AtomicOp::Swap(op) => wgsl::Instruction::AtomicSwap {
                lhs: self.compile_value(op.ptr),
                rhs: self.compile_value(op.value),
                out: self.compile_value(out.unwrap()),
            },
            cube::AtomicOp::CompareAndSwap(op) => wgsl::Instruction::AtomicCompareExchangeWeak {
                ptr: self.compile_value(op.ptr),
                cmp: self.compile_value(op.cmp),
                value: self.compile_value(op.val),
                out: self.compile_value(out.unwrap()),
            },
        }
    }

    fn compile_binding(&mut self, arg: kernel::KernelArg) -> wgsl::KernelArg {
        wgsl::KernelArg {
            id: arg.id,
            visibility: self.buffer_vis[arg.id as usize],
            value: self.compile_value(arg.value),
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
                register_extension(wgsl::Extension::PowfPrimitive(out.elem()));
                register_extension(wgsl::powf_extension(rhs, out));
            }
            #[cfg(target_os = "macos")]
            wgsl::Instruction::Tanh { input, out: _ } => {
                register_extension(wgsl::Extension::SafeTanhPrimitive(input.elem()));
                register_extension(wgsl::Extension::SafeTanh(input.item()));
            }
            wgsl::Instruction::IsNan { input, out } => {
                register_extension(wgsl::Extension::IsNanPrimitive(input.elem()));
                register_extension(wgsl::Extension::IsNan(input.item(), out.item()));
            }
            wgsl::Instruction::IsInf { input, out } => {
                register_extension(wgsl::Extension::IsInfPrimitive(input.elem()));
                register_extension(wgsl::Extension::IsInf(input.item(), out.item()));
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
