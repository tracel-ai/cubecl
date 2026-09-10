use cubecl_ir::{
    AddressType, ElemType, Scope, UIntKind,
    metadata::Info,
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, CubeKernel, KernelDefinition, KernelMetadata};

#[derive(Clone, Debug, Default)]
struct TestCompiler;

impl Compiler for TestCompiler {
    type Representation = String;
    type CompilationOptions = ();

    fn compile(
        &mut self,
        _kernel: KernelDefinition,
        _options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        Ok("void main() {}".to_string())
    }

    fn extension(&self) -> &'static str {
        "test"
    }

    fn lang_tag(&self) -> &'static str {
        "test"
    }
}

struct ComptimeKernel {
    strategy: &'static str,
}

impl KernelMetadata for ComptimeKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.strategy)
    }

    fn address_type(&self) -> ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}

impl CubeKernel for ComptimeKernel {
    fn define(&self) -> KernelDefinition {
        let mut settings =
            KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32);
        settings.kernel_name = "reduce_kernel_in_f32_out_u32".to_string();
        KernelDefinition {
            body: Scope::root(settings.clone()),
            settings,
            info: Info::default(),
        }
    }
}

#[test]
fn entrypoint_name_disambiguates_comptime_variants() {
    let k_argmax = ComptimeKernel { strategy: "ArgMax" };
    let k_argmin = ComptimeKernel { strategy: "ArgMin" };

    let mut compiler = TestCompiler;

    let c_max = CompiledKernel::compile(&k_argmax, k_argmax.define(), &mut compiler, &()).unwrap();
    let c_min = CompiledKernel::compile(&k_argmin, k_argmin.define(), &mut compiler, &()).unwrap();

    assert_ne!(
        c_max.entrypoint_name, c_min.entrypoint_name,
        "Kernels with distinct KernelId info must have distinct entrypoint names"
    );
    assert!(
        c_max
            .entrypoint_name
            .starts_with("reduce_kernel_in_f32_out_u32_")
    );
    assert!(
        c_min
            .entrypoint_name
            .starts_with("reduce_kernel_in_f32_out_u32_")
    );
}
