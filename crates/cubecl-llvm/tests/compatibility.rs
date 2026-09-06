//! Exercise the public API as a downstream crate, including CPU-only builds.
use cubecl_core::{Compiler, ir::amd::GfxArch};
use cubecl_environment::bytes::Bytes;
use cubecl_llvm::{AmdGpuModule, LlvmTarget, PlironArtifact, PlironCompiler, PlironOptions};

#[test]
fn public_options_targets_and_artifacts_remain_available() {
    let options = PlironOptions { arch: None };
    assert!(options.arch.is_none());
    let gpu_options = PlironOptions {
        arch: Some(GfxArch::parse("gfx1201")),
    };
    assert_eq!(gpu_options.arch.unwrap().name(), "gfx1201");

    // Callers can still match every target and artifact, even when they only
    // execute CPU kernels or inspect artifacts compiled by another process.
    for target in [LlvmTarget::Cpu, LlvmTarget::AmdGpu] {
        let expected = match target {
            LlvmTarget::Cpu => "plir",
            LlvmTarget::AmdGpu => "ll",
        };
        assert_eq!(PlironCompiler { target }.extension(), expected);
    }
    let artifact = PlironArtifact::AmdGpuCode(AmdGpuModule {
        code_object: Bytes::from_bytes_vec(vec![]),
        entrypoint: "k".into(),
        ir: "define void @k() { ret void }".into(),
        asm: None,
        shared_memory_size: 0,
        io: vec![],
    });
    match &artifact {
        PlironArtifact::Jit(_) => panic!("expected an AMDGPU artifact"),
        PlironArtifact::AmdGpuCode(module) => assert_eq!(module.entrypoint, "k"),
    }
    assert!(artifact.to_string().contains("@k"));
}

#[cfg(not(feature = "amdgpu"))]
#[test]
fn unavailable_amdgpu_returns_a_compilation_error() {
    use cubecl_core::CompilationError;

    for arch in [None, Some(GfxArch::parse("gfx1201"))] {
        let kernel = empty_kernel();
        let mut compiler = PlironCompiler {
            target: LlvmTarget::AmdGpu,
        };
        match compiler.compile(kernel, &PlironOptions { arch }) {
            Err(CompilationError::Generic { reason, .. }) => {
                assert!(reason.contains("cubecl-llvm/amdgpu"), "{reason}");
            }
            _ => panic!("a disabled AMDGPU compiler must report the missing feature"),
        }
    }
}

#[cfg(not(feature = "amdgpu"))]
#[test]
fn direct_amdgpu_codegen_and_linking_report_the_missing_feature() {
    use cubecl_llvm::amdgpu::{codegen::emit_code_object, lld::link_relocatable};
    use pliron::printable::Printable;

    let kernel = empty_kernel();
    let module = kernel.body.state().module;
    let ctx = kernel.body.ctx();
    let before = module.disp(ctx).to_string();
    let result = emit_code_object(ctx, module, "k", &GfxArch::parse("gfx1201"), 1, 0, vec![]);
    let error = result.unwrap_err();
    assert!(error.contains("cubecl-llvm/amdgpu"), "{error}");
    assert_eq!(module.disp(ctx).to_string(), before);
    let error = link_relocatable(&[], "k").unwrap_err();
    assert!(error.contains("cubecl-llvm/amdgpu"), "{error}");
}

#[cfg(not(feature = "amdgpu"))]
fn empty_kernel() -> cubecl_core::prelude::KernelDefinition {
    use cubecl_core::{
        compute::KernelBuilder,
        ir::{
            AddressType,
            settings::{Dim3, ExecutionMode, KernelSettings},
        },
    };
    KernelBuilder::new(
        KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32)
            .kernel_name("k"),
    )
    .build()
}
