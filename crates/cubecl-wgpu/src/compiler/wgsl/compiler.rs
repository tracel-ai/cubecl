use super::shader::ComputeShader;
use crate::compiler::wgsl::{
    self, EnableFeaturesPass, builtin::LowerBuiltinsPass, lower::LowerOpsWgslPass,
    metadata::declare_info, rewrite_args, shared_memory_size,
};

use cubecl_core::{
    WgpuCompilationOptions,
    backtrace::BackTrace,
    post_processing::{
        disaggregate::DisaggregatePass, saturating::LowerSaturatingArithmeticPass,
        unroll::UnrollPass,
    },
};
use cubecl_ir::{
    ContextExt,
    pliron::{
        builtin::ops::{FuncOp, ModuleOp},
        operation::verify_operation,
        opts::{constants::sccp::SCCPPass, dce::DCEPass, mem2reg::Mem2RegPass},
        printable::Printable,
    },
    prelude::{AnalysisManager, NestedOpsPass, Op, OpPass, PMConfig, Pass, Passes},
    rewrite::SimplifyOpsPass,
    settings::Dim3,
};
use cubecl_opt::passes::{
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, simple_cse::SimpleCSEPass,
};
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::kernel;

const MAX_VECTOR_SIZE: usize = 4;

pub struct KernelInfo {
    pub cube_dim: Dim3,
}

/// Wgsl Compiler.
#[derive(Clone, Default)]
pub struct WgslCompiler;

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
    ) -> Result<Self::Representation, CompilationError> {
        self.compile_shader(shader, compilation_options)
    }

    fn extension(&self) -> &'static str {
        "wgsl"
    }
}

impl WgslCompiler {
    fn compile_shader(
        &mut self,
        value: kernel::KernelDefinition,
        _compilation_options: &WgpuCompilationOptions,
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

        let module = value.body.state().module;
        let entry_func = value.body.state().entry_func;
        let module_op = module.get_operation();
        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty(value.info);
        ctx.set_aux_ty(KernelInfo {
            cube_dim: value.settings.cube_dim,
        });

        std::fs::write("target/initial.plir", format!("{}", module.disp(&ctx))).unwrap();
        verify_operation(module_op, &ctx).expect("Failed to verify before passes");

        let config = PMConfig {
            print_after_all: true,
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        // func_passes.add_pass(LowerInfoPass);
        func_passes.add_pass(DisaggregatePass);
        func_passes.add_pass(UnrollPass::new(MAX_VECTOR_SIZE));

        func_passes.add_pass(LowerOpsWgslPass::default());

        func_passes.add_pass(LowerSaturatingArithmeticPass::default());

        func_passes.add_pass(LowerBuiltinsPass);

        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(DCEPass);

        // SCCP/DCE may unlock more mem2reg opportunities, and vice versa. So we do a sandwich.
        func_passes.add_pass(Mem2RegPass);

        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);
        passes.add_pass(EnableFeaturesPass);

        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        let buffers = rewrite_args(&mut ctx, entry_func);
        declare_info(&mut ctx, module, buffers.len());
        let shared_memory_size = shared_memory_size(&ctx, module_op);

        std::fs::write(
            "target/after_lower_shared.plir",
            format!("{}", module.disp(&ctx)),
        )
        .unwrap();

        verify_operation(module.get_operation(), &ctx).expect("Failed to verify after passes");

        Ok(ComputeShader {
            buffers,
            shared_memory_size,
            ctx,
        })
    }

    // fn compile_operation(
    //     &mut self,
    //     instructions: &mut Vec<wgsl::Instruction>,
    //     operation: cube::Operation,
    //     out: Option<cube::ExpandValue>,
    //     scope: &cube::Scope,
    // ) {
    //     match operation {
    //         cube::Operation::DeclareVariable {
    //             value_ty,
    //             addr_space: AddressSpace::Local,
    //             ..
    //         } => instructions.push(wgsl::Instruction::DeclareVariable {
    //             val: self.compile_value(out.unwrap()),
    //             value_ty: self.compile_type(value_ty),
    //         }),
    //         cube::Operation::DeclareVariable {
    //             value_ty,
    //             addr_space: AddressSpace::Shared,
    //             alignment,
    //         } => {
    //             let ty = self.compile_type(value_ty);
    //             let value = self.compile_value(out.unwrap());
    //             self.shared_values
    //                 .push(SharedValue::new(ty, value, alignment as u32));
    //         }
    //         cube::Operation::DeclareVariable { addr_space, .. } => {
    //             unimplemented!("Unsupported declare address space {addr_space}")
    //         }
    //         cube::Operation::Memory(memory) => self.compile_memory(memory, out, instructions),
    //         cube::Operation::Arithmetic(op) => {
    //             self.compile_arithmetic(op, out, instructions, scope)
    //         }
    //         cube::Operation::Comparison(op) => self.compile_cmp(op, out, instructions),
    //         cube::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
    //         cube::Operation::Operator(op) => self.compile_operator(op, out, instructions),
    //         cube::Operation::Atomic(op) => instructions.push(self.compile_atomic(op, out)),
    //         cube::Operation::Synchronization(val) => {
    //             self.compile_synchronization(instructions, val)
    //         }
    //         cube::Operation::WorkgroupUniformLoad(op) => {
    //             instructions.push(wgsl::Instruction::WorkgroupUniformLoad {
    //                 input: self.compile_value(op),
    //                 out: self.compile_value(out.unwrap()),
    //             });
    //         }
    //         cube::Operation::Plane(op) => self.compile_subgroup(instructions, op, out),
    //         cube::Operation::CoopMma(_) => {
    //             panic!("Cooperative matrix-multiply and accumulate isn't supported on wgpu.")
    //         }
    //         cube::Operation::NonSemantic(cube::NonSemantic::Comment { content }) => {
    //             self.compile_comment(instructions, content)
    //         }
    //         cube::Operation::NonSemantic(_) => {}
    //         cube::Operation::Barrier(_) => {
    //             panic!("Barrier isn't supported on wgpu.")
    //         }
    //         cube::Operation::Tma(_) => panic!("TMA isn't supported on wgpu."),
    //         cube::Operation::TensorIndexing(_) => panic!("TMA isn't supported on wgpu."),
    //         cube::Operation::Marker(_) => {}
    //         cube::Operation::ConstructAggregate(..)
    //         | cube::Operation::ExtractAggregateField(..) => {
    //             unreachable!("Should be disaggregated at this point")
    //         }
    //     }
    // }

    // fn compile_subgroup(
    //     &mut self,
    //     instructions: &mut Vec<wgsl::Instruction>,
    //     subgroup: cube::Plane,
    //     out: Option<cube::ExpandValue>,
    // ) {
    //     self.subgroup_instructions_used = true;

    //     let out = out.unwrap();
    //     let op = match subgroup {
    //         cube::Plane::Elect => Subgroup::Elect {
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::All(op) => Subgroup::All {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::Any(op) => Subgroup::Any {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::Ballot(op) => Subgroup::Ballot {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },

    //         cube::Plane::Broadcast(op) => Subgroup::Broadcast {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         },

    //         cube::Plane::Sum(op) => Subgroup::Sum {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },

    //         cube::Plane::ExclusiveSum(op) => Subgroup::ExclusiveSum {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::InclusiveSum(op) => Subgroup::InclusiveSum {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::Prod(op) => Subgroup::Prod {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::ExclusiveProd(op) => Subgroup::ExclusiveProd {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::InclusiveProd(op) => Subgroup::InclusiveProd {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::Min(op) => Subgroup::Min {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::Max(op) => Subgroup::Max {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::Shuffle(op) => Subgroup::Shuffle {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::ShuffleXor(op) => Subgroup::ShuffleXor {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::ShuffleUp(op) => Subgroup::ShuffleUp {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         },
    //         cube::Plane::ShuffleDown(op) => Subgroup::ShuffleDown {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         },
    //     };

    //     instructions.push(wgsl::Instruction::Subgroup(op));
    // }

    // fn compile_synchronization(
    //     &mut self,
    //     instructions: &mut Vec<wgsl::Instruction>,
    //     synchronization: cube::Synchronization,
    // ) {
    //     match synchronization {
    //         cube::Synchronization::SyncCube => {
    //             instructions.push(wgsl::Instruction::WorkgroupBarrier)
    //         }
    //         cube::Synchronization::SyncPlane => {
    //             panic!("Synchronization within a plane is not supported in WGSL")
    //         }
    //         cube::Synchronization::SyncStorage => {
    //             instructions.push(wgsl::Instruction::StorageBarrier)
    //         }
    //         cube::Synchronization::SyncAsyncProxyShared => panic!("TMA is not supported in WGSL"),
    //     };
    // }

    // fn compile_comment(&mut self, instructions: &mut Vec<wgsl::Instruction>, content: String) {
    //     instructions.push(wgsl::Instruction::Comment { content })
    // }

    // fn compile_memory(
    //     &mut self,
    //     value: cube::Memory,
    //     out: Option<cube::ExpandValue>,
    //     instructions: &mut Vec<wgsl::Instruction>,
    // ) {
    //     match value {
    //         cube::Memory::Index(op) => {
    //             instructions.push(wgsl::Instruction::Index {
    //                 lhs: self.compile_value(op.list),
    //                 rhs: self.compile_value(op.index),
    //                 out: self.compile_value(out.unwrap()),
    //             });
    //         }
    //         cube::Memory::Load(value) => instructions.push(wgsl::Instruction::Load {
    //             input: self.compile_value(value),
    //             out: self.compile_value(out.unwrap()),
    //         }),
    //         cube::Memory::Store(op) => instructions.push(wgsl::Instruction::Store {
    //             input: self.compile_value(op.value),
    //             out: self.compile_value(op.ptr),
    //         }),
    //         cube::Memory::CopyMemory(op) => instructions.push(wgsl::Instruction::CopyBulk {
    //             source: self.compile_value(op.source),
    //             target: self.compile_value(op.target),
    //             len: op.len as u32,
    //         }),
    //     }
    // }

    // fn compile_arithmetic(
    //     &mut self,
    //     value: cube::Arithmetic,
    //     out: Option<cube::ExpandValue>,
    //     instructions: &mut Vec<wgsl::Instruction>,
    //     scope: &Scope,
    // ) {
    //     let out = out.unwrap();
    //     match value {
    //         cube::Arithmetic::Magnitude(op) => instructions.push(wgsl::Instruction::Magnitude {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Arithmetic::Normalize(op) => instructions.push(wgsl::Instruction::Normalize {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Arithmetic::Dot(op) => instructions.push(wgsl::Instruction::Dot {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Arithmetic::VectorSum(op) => instructions.push(wgsl::Instruction::VectorSum {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //     }
    // }

    // fn compile_cmp(
    //     &mut self,
    //     value: cube::Comparison,
    //     out: Option<cube::ExpandValue>,
    //     instructions: &mut Vec<wgsl::Instruction>,
    // ) {
    //     let out = out.unwrap();
    //     match value {
    //         cube::Comparison::IsNan(op) => instructions.push(wgsl::Instruction::IsNan {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Comparison::IsInf(op) => instructions.push(wgsl::Instruction::IsInf {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //     }
    // }

    // fn compile_bitwise(
    //     &mut self,
    //     value: cube::Bitwise,
    //     out: Option<cube::ExpandValue>,
    //     instructions: &mut Vec<wgsl::Instruction>,
    // ) {
    //     let out = out.unwrap();
    //     match value {
    //         cube::Bitwise::BitwiseOr(op) => instructions.push(wgsl::Instruction::BitwiseOr {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::BitwiseAnd(op) => instructions.push(wgsl::Instruction::BitwiseAnd {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::BitwiseXor(op) => instructions.push(wgsl::Instruction::BitwiseXor {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::CountOnes(op) => instructions.push(wgsl::Instruction::CountBits {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::ReverseBits(op) => instructions.push(wgsl::Instruction::ReverseBits {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::ShiftLeft(op) => instructions.push(wgsl::Instruction::ShiftLeft {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::ShiftRight(op) => instructions.push(wgsl::Instruction::ShiftRight {
    //             lhs: self.compile_value(op.lhs),
    //             rhs: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::BitwiseNot(op) => instructions.push(wgsl::Instruction::BitwiseNot {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::LeadingZeros(op) => instructions.push(wgsl::Instruction::LeadingZeros {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Bitwise::TrailingZeros(op) => {
    //             instructions.push(wgsl::Instruction::TrailingZeros {
    //                 input: self.compile_value(op.input),
    //                 out: self.compile_value(out),
    //             })
    //         }
    //         cube::Bitwise::FindFirstSet(op) => instructions.push(wgsl::Instruction::FindFirstSet {
    //             input: self.compile_value(op.input),
    //             out: self.compile_value(out),
    //         }),
    //     }
    // }

    // fn compile_operator(
    //     &mut self,
    //     value: cube::Operator,
    //     out: Option<cube::ExpandValue>,
    //     instructions: &mut Vec<wgsl::Instruction>,
    // ) {
    //     let out = out.unwrap();
    //     match value {
    //         cube::Operator::InitVector(op) => instructions.push(wgsl::Instruction::VecInit {
    //             inputs: op
    //                 .inputs
    //                 .into_iter()
    //                 .map(|val| self.compile_value(val))
    //                 .collect(),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Operator::ExtractComponent(op) => instructions.push(wgsl::Instruction::Extract {
    //             vector: self.compile_value(op.lhs),
    //             index: self.compile_value(op.rhs),
    //             out: self.compile_value(out),
    //         }),
    //         cube::Operator::InsertComponent(op) => instructions.push(wgsl::Instruction::Insert {
    //             vector: self.compile_value(op.vector),
    //             index: self.compile_value(op.index),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out),
    //         }),
    //     }
    // }

    // fn compile_atomic(
    //     &mut self,
    //     atomic: cube::AtomicOp,
    //     out: Option<cube::ExpandValue>,
    // ) -> wgsl::Instruction {
    //     match atomic {
    //         cube::AtomicOp::Add(op) => wgsl::Instruction::AtomicAdd {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Sub(op) => wgsl::Instruction::AtomicSub {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Max(op) => wgsl::Instruction::AtomicMax {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Min(op) => wgsl::Instruction::AtomicMin {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::And(op) => wgsl::Instruction::AtomicAnd {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Or(op) => wgsl::Instruction::AtomicOr {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Xor(op) => wgsl::Instruction::AtomicXor {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Load(ptr) => wgsl::Instruction::AtomicLoad {
    //             input: self.compile_value(ptr),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Store(op) => wgsl::Instruction::AtomicStore {
    //             input: self.compile_value(op.value),
    //             out: self.compile_value(op.ptr),
    //         },
    //         cube::AtomicOp::Swap(op) => wgsl::Instruction::AtomicSwap {
    //             lhs: self.compile_value(op.ptr),
    //             rhs: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::CompareAndSwap(op) => wgsl::Instruction::AtomicCompareExchangeWeak {
    //             ptr: self.compile_value(op.ptr),
    //             cmp: self.compile_value(op.cmp),
    //             value: self.compile_value(op.val),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //     }
    // }
}

// fn register_extensions(instructions: &[wgsl::Instruction]) -> Vec<wgsl::Extension> {
//     let mut extensions = Vec::new();

//     let mut register_extension = |extension: wgsl::Extension| {
//         if !extensions.contains(&extension) {
//             extensions.push(extension);
//         }
//     };

//     // Since not all instructions are native to WGSL, we need to add the custom ones.
//     for instruction in instructions {
//         match instruction {
//             wgsl::Instruction::Powf { lhs: _, rhs, out } => {
//                 register_extension(wgsl::Extension::PowfPrimitive(out.elem()));
//                 register_extension(wgsl::powf_extension(rhs, out));
//             }
//             #[cfg(target_os = "macos")]
//             wgsl::Instruction::Tanh { input, out: _ } => {
//                 register_extension(wgsl::Extension::SafeTanhPrimitive(input.elem()));
//                 register_extension(wgsl::Extension::SafeTanh(input.item()));
//             }
//             wgsl::Instruction::IsNan { input, out } => {
//                 register_extension(wgsl::Extension::IsNanPrimitive(input.elem()));
//                 register_extension(wgsl::Extension::IsNan(input.item(), out.item()));
//             }
//             wgsl::Instruction::IsInf { input, out } => {
//                 register_extension(wgsl::Extension::IsInfPrimitive(input.elem()));
//                 register_extension(wgsl::Extension::IsInf(input.item(), out.item()));
//             }
//             wgsl::Instruction::If { instructions, .. } => {
//                 for extension in register_extensions(instructions) {
//                     register_extension(extension);
//                 }
//             }
//             wgsl::Instruction::IfElse {
//                 instructions_if,
//                 instructions_else,
//                 ..
//             } => {
//                 for extension in register_extensions(instructions_if) {
//                     register_extension(extension);
//                 }
//                 for extension in register_extensions(instructions_else) {
//                     register_extension(extension);
//                 }
//             }
//             wgsl::Instruction::Loop { instructions } => {
//                 for extension in register_extensions(instructions) {
//                     register_extension(extension);
//                 }
//             }
//             wgsl::Instruction::RangeLoop { instructions, .. } => {
//                 for extension in register_extensions(instructions) {
//                     register_extension(extension);
//                 }
//             }
//             _ => {}
//         }
//     }

//     extensions
// }
