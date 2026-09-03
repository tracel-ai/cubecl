//! A kernel carrying its own compiled text skips the compiler, and is only
//! accepted by a compiler of its language.

use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::{
    AddressType, ElemType, Scope, UIntKind,
    metadata::Info,
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{
    CompiledKernel, CubeKernel, KernelDefinition, KernelMetadata, PrecompiledSource,
};

const SOURCE: &str = "fn main() {}";

/// A compiler that never compiles: the tag is all a precompiled kernel asks of it.
#[derive(Clone, Debug)]
struct TaggedCompiler(&'static str);

impl Compiler for TaggedCompiler {
    type Representation = String;
    type CompilationOptions = ();

    fn compile(
        &mut self,
        _kernel: KernelDefinition,
        _options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        Err(CompilationError::Generic {
            reason: "a precompiled kernel must not reach the compiler".to_string(),
            backtrace: BackTrace::capture(),
        })
    }

    fn extension(&self) -> &'static str {
        self.0
    }

    fn lang_tag(&self) -> &'static str {
        self.0
    }
}

/// A kernel that brings its own text, tagged with `lang`.
struct HandWritten {
    lang: &'static str,
}

impl KernelMetadata for HandWritten {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.lang)
    }

    fn address_type(&self) -> ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}

impl CubeKernel for HandWritten {
    fn define(&self) -> KernelDefinition {
        let settings =
            KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32);
        KernelDefinition {
            body: Scope::root(settings.clone()),
            settings,
            info: Info::default(),
        }
    }

    fn source(&self) -> Option<PrecompiledSource> {
        Some(PrecompiledSource {
            source: SOURCE.to_string(),
            entrypoint_name: "main".to_string(),
            lang: self.lang,
        })
    }
}

fn compile(
    kernel_lang: &'static str,
    compiler_lang: &'static str,
) -> Result<CompiledKernel<TaggedCompiler>, CompilationError> {
    let kernel = HandWritten { lang: kernel_lang };
    let definition = kernel.define();
    CompiledKernel::compile(&kernel, definition, &mut TaggedCompiler(compiler_lang), &())
}

#[test]
fn a_kernel_in_the_compilers_language_passes_through() {
    let compiled = compile("wgsl", "wgsl").expect("the tags match");

    assert_eq!(compiled.source, SOURCE);
    assert_eq!(compiled.entrypoint_name, "main");
    assert!(
        compiled.repr.is_none(),
        "there is no representation to keep"
    );
    assert!(compiled.io.is_none(), "every buffer reads as read-write");
}

#[test]
fn a_kernel_in_another_language_is_refused() {
    let reason = match compile("cuda", "wgsl") {
        Ok(_) => panic!("the tags differ, the kernel must be refused"),
        Err(CompilationError::Generic { reason, .. }) => reason,
        Err(err) => panic!("expected a generic compilation error, got {err}"),
    };
    assert!(
        reason.contains("cuda"),
        "names the kernel's language: {reason}"
    );
    assert!(
        reason.contains("wgsl"),
        "names the compiler's language: {reason}"
    );
}
