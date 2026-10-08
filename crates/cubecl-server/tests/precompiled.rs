//! A kernel carrying its own compiled text skips the compiler, and is only
//! accepted by a compiler of its language. One carrying a binary is only
//! accepted by a compiler that loads binaries.

use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::bytes::Bytes;
use cubecl_ir::{
    AddressType, ElemType, Scope, UIntKind,
    metadata::Info,
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_server::compiler::{CompilationError, Compiler};
use cubecl_server::id::KernelId;
use cubecl_server::kernel::{
    CompiledKernel, CubeKernel, KernelDefinition, KernelMetadata, KernelParam, PrecompiledBinary,
    PrecompiledSource,
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

/// A compiler that loads binaries, keeping what it was handed.
#[derive(Clone, Debug)]
struct LoadingCompiler;

impl Compiler for LoadingCompiler {
    type Representation = String;
    type CompilationOptions = ();

    fn compile(
        &mut self,
        _kernel: KernelDefinition,
        _options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        Err(CompilationError::Generic {
            reason: "a precompiled binary must not reach the compiler".to_string(),
            backtrace: BackTrace::capture(),
        })
    }

    fn extension(&self) -> &'static str {
        "bin"
    }

    fn lang_tag(&self) -> &'static str {
        "bin"
    }

    fn load_binary(
        &mut self,
        binary: PrecompiledBinary,
    ) -> Result<Self::Representation, CompilationError> {
        Ok(format!("{} {:?}", binary.entrypoint_name, binary.params))
    }
}

/// A kernel that brings its own module image, and optionally text too.
struct Foreign {
    with_source: bool,
}

impl KernelMetadata for Foreign {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.with_source)
    }

    fn address_type(&self) -> ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}

impl CubeKernel for Foreign {
    fn define(&self) -> KernelDefinition {
        HandWritten { lang: "bin" }.define()
    }

    fn source(&self) -> Option<PrecompiledSource> {
        self.with_source.then(|| PrecompiledSource {
            source: SOURCE.to_string(),
            entrypoint_name: "main".to_string(),
            lang: "bin",
        })
    }

    fn binary(&self) -> Option<PrecompiledBinary> {
        Some(PrecompiledBinary {
            image: Bytes::from_bytes_vec(vec![0x7f, b'E', b'L', b'F']),
            entrypoint_name: "main".to_string(),
            params: vec![KernelParam::Resource(0), KernelParam::Info(0)],
        })
    }
}

fn reason(result: Result<CompiledKernel<impl Compiler>, CompilationError>) -> String {
    match result {
        Ok(_) => panic!("the kernel must be refused"),
        Err(CompilationError::Generic { reason, .. }) => reason,
        Err(err) => panic!("expected a generic compilation error, got {err}"),
    }
}

#[test]
fn a_binary_reaches_a_compiler_that_loads_it() {
    let kernel = Foreign { with_source: false };
    let compiled = CompiledKernel::compile(&kernel, kernel.define(), &mut LoadingCompiler, &())
        .expect("the compiler loads binaries");

    assert_eq!(compiled.entrypoint_name, "main");
    assert_eq!(
        compiled.repr.as_deref(),
        Some("main [Resource(0), Info(0)]"),
        "the compiler was handed the binary's parameter list"
    );
    assert!(compiled.io.is_none(), "every buffer reads as read-write");
}

#[test]
fn a_binary_is_refused_by_a_compiler_that_does_not_load_one() {
    let kernel = Foreign { with_source: false };
    let reason = reason(CompiledKernel::compile(
        &kernel,
        kernel.define(),
        &mut TaggedCompiler("wgsl"),
        &(),
    ));
    assert!(reason.contains("main"), "names the entrypoint: {reason}");
}

#[test]
fn a_kernel_with_both_a_binary_and_a_source_is_refused() {
    let kernel = Foreign { with_source: true };
    let reason = reason(CompiledKernel::compile(
        &kernel,
        kernel.define(),
        &mut LoadingCompiler,
        &(),
    ));
    assert!(reason.contains("both"), "says why: {reason}");
}
