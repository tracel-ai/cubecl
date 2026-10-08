//! Debug data in the SPIR-V of a kernel.
//!
//! `pliron-spirv` converts the location of each op to debug data. The format is core `OpLine`, or
//! `NonSemantic.Shader.DebugInfo.100` with each inlined `#[cube]` function as a separate frame.
//! [`debug_format`] selects the format from the configuration and the device. `pliron-spirv` owns the placement rules, for example no instruction between a merge
//! instruction and its branch.

use cubecl_core::WgpuCompilationOptions;
use cubecl_ir::{ContextExt, debug::DebugState, settings::DebugInfo};
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, compilation::SpirvDebugFormat};
use pliron::context::Context;
use pliron_spirv::{
    PlironBuilder,
    debug_info::{DebugInfoFormat, DebugInfoOptions},
};
use rspirv::spirv::SourceLanguage;

/// The builder of the module of a kernel with the debug data `level`. At [`DebugInfo::None`], the
/// module has no debug data.
pub(crate) fn builder(ctx: &Context, level: DebugInfo) -> PlironBuilder {
    if level == DebugInfo::None {
        return PlironBuilder::default();
    }
    let supported = ctx
        .aux_ty::<WgpuCompilationOptions>()
        .vulkan
        .supports_non_semantic_info;
    let configured = CubeClRuntimeConfig::get().compilation.spirv_debug_format;
    let mut options = DebugInfoOptions::default();
    options.format = debug_format(configured, supported);
    options.language = SourceLanguage::Rust;
    options.producer = "cubecl".to_string();
    if let Some(debug) = ctx.try_aux_ty::<DebugState>() {
        // The directory that has the relative source files on this computer, if one does. At
        // `Full`, the file must have the compiled text.
        #[cfg(feature = "std")]
        {
            use cubecl_runtime::debug_source::{kernel_source_root, source_md5s};
            if let Some(root) = kernel_source_root(debug, &source_md5s(debug, level)) {
                options.directory = root;
            }
        }
        // Only `Full` embeds the source text, as the macro records it only then.
        if level == DebugInfo::Full {
            options.source_text = debug
                .sources()
                .iter()
                .map(|(path, text)| (path.clone(), text.to_string()))
                .collect();
        }
    }
    PlironBuilder::with_debug_info(options)
}

/// The format for the configured format `configured`, on a device that `supported`
/// `NonSemantic.Shader.DebugInfo.100` or not. A device without support gets `OpLine`.
fn debug_format(configured: SpirvDebugFormat, supported: bool) -> DebugInfoFormat {
    match configured {
        SpirvDebugFormat::Auto | SpirvDebugFormat::NonSemantic if supported => {
            DebugInfoFormat::NonSemantic
        }
        SpirvDebugFormat::Auto | SpirvDebugFormat::OpLine => DebugInfoFormat::OpLine,
        SpirvDebugFormat::NonSemantic => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                log::warn!(
                    "The device does not support NonSemantic.Shader.DebugInfo.100. \
                     SPIR-V kernels get OpLine debug data."
                );
            });
            DebugInfoFormat::OpLine
        }
    }
}

#[cfg(test)]
// A `#[cube]` kernel loops over a range, not over an iterator.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::debug_format;
    use crate::SpirvCompiler;
    use cubecl_core as cubecl;
    use cubecl_core::{
        Compiler, VulkanCompilationOptions, WgpuCompilationOptions,
        ir::{
            DeviceProperties, ElemType, FloatKind, UIntKind, features::TypeUsage,
            settings::DebugInfo,
        },
        prelude::*,
        runtime_tests::offline::{
            SOURCE, SOURCE_PATH, doubled, nested_calls_with_source_kernel,
            offline_device_properties, source_line,
        },
    };
    use cubecl_runtime::config::compilation::SpirvDebugFormat;
    use cubecl_runtime::kernel::CubeKernel;
    use pliron_spirv::debug_info::DebugInfoFormat;
    use rspirv::{
        binary::Disassemble,
        dr::{Instruction, Module, Operand},
        spirv::{DebugInfoOp, Op as SpirvOp},
    };
    use std::{
        io::{ErrorKind, Write},
        process::{Command, Stdio},
        sync::Arc,
    };

    /// The branches and the loop give selection merges and a loop merge, which no `OpLine` may
    /// separate from their branches.
    #[cube(launch)]
    fn outer(input: &[f32], output: &mut [f32]) {
        if ABSOLUTE_POS < input.len() {
            let mut acc = 0.0;
            for i in 0..input.len() {
                if i != ABSOLUTE_POS {
                    acc += doubled(input[i]);
                }
            }
            output[ABSOLUTE_POS] = acc;
        }
    }

    fn properties() -> Arc<DeviceProperties> {
        let mut properties = offline_device_properties(32);
        properties.register_address_type(AddressType::U32);
        for ty in [
            ElemType::Index,
            ElemType::UInt(UIntKind::U32),
            ElemType::Float(FloatKind::F32),
            ElemType::Bool,
        ] {
            properties.register_type_usage(ty, TypeUsage::all());
        }
        Arc::new(properties)
    }

    /// Compiles `kernel` at the debug level `level`, on a device that supports
    /// `NonSemantic.Shader.DebugInfo.100` or not. The level is set after the resolution, which
    /// gives a `dev` build at least line tables, as `CUBECL_DEBUG_INFO` does.
    fn compile(kernel: &impl CubeKernel, level: DebugInfo, non_semantic: bool) -> Module {
        let mut definition = kernel.define();
        definition.settings.debug_info = level;
        let options = WgpuCompilationOptions {
            supports_u64: true,
            supports_vulkan_compiler: true,
            supports_msl_compiler: false,
            vulkan: VulkanCompilationOptions {
                max_spirv_version: (1, 6),
                max_vector_size: 4,
                supports_non_semantic_info: non_semantic,
                ..Default::default()
            },
        };
        let kernel = SpirvCompiler.compile(definition, &options).unwrap();
        let module = Arc::unwrap_or_clone(kernel.module.unwrap());
        validate(&kernel.assembled_module);
        module
    }

    /// The SPIR-V of `outer`.
    fn compile_outer(level: DebugInfo, non_semantic: bool) -> Module {
        let kernel = outer::Outer::new(
            KernelSettings::new(
                *CubeDim::new_1d(64),
                ExecutionMode::Checked,
                AddressType::U32,
            ),
            properties(),
            Arc::new(TargetProperties::default()),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        compile(&kernel, level, non_semantic)
    }

    /// Runs `spirv-val` on `words`. Without `spirv-val`, does nothing, except in CI, where the
    /// workflow installs it.
    fn validate(words: &[u32]) {
        let mut child = match Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3", "-"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) if err.kind() == ErrorKind::NotFound && std::env::var_os("CI").is_none() => {
                return;
            }
            Err(err) => panic!("spirv-val: {err}"),
        };
        let bytes = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "spirv-val: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// The text of the `OpString` with the id `id`.
    fn string(module: &Module, id: u32) -> &str {
        let string = module
            .debug_string_source
            .iter()
            .find(|it| it.result_id == Some(id))
            .unwrap();
        match &string.operands[0] {
            Operand::LiteralString(text) => text,
            operand => panic!("%{id} is not a string: {operand:?}"),
        }
    }

    /// The file and the line of each `OpLine` in the functions of `module`.
    fn op_lines(module: &Module) -> Vec<(&str, u32)> {
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == SpirvOp::Line)
            .map(|inst| match inst.operands[..] {
                [Operand::IdRef(file), Operand::LiteralBit32(line), ..] => {
                    (string(module, file), line)
                }
                _ => panic!("{inst:?}"),
            })
            .collect()
    }

    /// Without device support, each op gets the `OpLine` of its innermost `#[cube]` frame. The
    /// file name is the absolute path: the search finds the workspace root, a parent of the
    /// working directory of the test. `third * 2.0` is not checked: it fuses with `acc +=` into
    /// one `Fma`, which has the line of the addition.
    #[test]
    fn source_lines_are_op_lines() {
        let module = compile_outer(DebugInfo::LineTables, false);
        let lines = op_lines(&module);
        for text in ["let y = x * x;", "y / 3.0"] {
            let line = source_line(text);
            let (file, _) = lines
                .iter()
                .find(|(file, number)| file.ends_with(SOURCE_PATH) && *number == line)
                .unwrap_or_else(|| panic!("no `OpLine` for `{text}` in:\n{lines:?}"));
            let path = std::path::Path::new(file);
            assert!(path.is_absolute() && path.is_file(), "{file}");
        }
        assert!(!module.disassemble().contains("NonSemantic"));
    }

    /// The extended instructions `op` of `NonSemantic.Shader.DebugInfo.100` in `module`.
    fn debug_instructions(module: &Module, op: DebugInfoOp) -> Vec<&Instruction> {
        let functions = module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions);
        module
            .types_global_values
            .iter()
            .chain(functions)
            .filter(|inst| {
                inst.class.opcode == SpirvOp::ExtInst
                    && inst.operands[1] == Operand::LiteralExtInstInteger(op as u32)
            })
            .collect()
    }

    /// Without debug data, the module has no debug instruction, although the ops have locations.
    #[test]
    fn no_debug_data_without_level() {
        for non_semantic in [false, true] {
            let module = compile_outer(DebugInfo::None, non_semantic).disassemble();
            assert!(!module.contains("OpLine"), "{module}");
            assert!(!module.contains("NonSemantic"), "{module}");
        }
    }

    #[test]
    fn format_follows_configuration_and_device() {
        use SpirvDebugFormat::*;
        let cases = [
            (Auto, true, DebugInfoFormat::NonSemantic),
            (Auto, false, DebugInfoFormat::OpLine),
            (OpLine, true, DebugInfoFormat::OpLine),
            (OpLine, false, DebugInfoFormat::OpLine),
            (NonSemantic, true, DebugInfoFormat::NonSemantic),
            (NonSemantic, false, DebugInfoFormat::OpLine),
        ];
        for (configured, supported, expected) in cases {
            assert_eq!(debug_format(configured, supported), expected);
        }
    }

    /// At `Full`, the `DebugSource` of the kernel file has the text that the macro records.
    /// `spirv-val` checks each column against the length of its line in the text, so the columns
    /// of the macro must fit the lines.
    #[test]
    fn full_debug_data_embeds_the_source() {
        let module = compile(
            &nested_calls_with_source_kernel(properties()),
            DebugInfo::Full,
            true,
        );
        let texts = debug_instructions(&module, DebugInfoOp::DebugSource)
            .into_iter()
            .filter_map(|inst| match inst.operands[2..] {
                [_, Operand::IdRef(text), ..] => Some(string(&module, text)),
                _ => None,
            })
            .collect::<Vec<_>>();
        // A long text continues in `DebugSourceContinued`, so the first part is a prefix.
        let [text] = texts[..] else {
            panic!("{}", module.disassemble());
        };
        assert!(!text.is_empty() && SOURCE.starts_with(text), "{text}");
    }
}
