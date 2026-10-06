//! Kernel source locations as LLVM debug data.
//!
//! `pliron-llvm` converts the `Location` of each op to a `!dbg` location: one `DISubprogram` for
//! each function, and one for each inlined `#[cube]` function.

use crate::{prelude::*, shared::llvm_module::LlvmModule};
use cubecl_core::ir::{ContextExt, debug::DebugState, settings::DebugInfo};
use cubecl_runtime::debug_source::{kernel_source_root, source_md5s};
use pliron_llvm::{
    debug_info_conversions::to_llvm_ir::{
        DebugInfoOptions, EmissionKind, LLVMDWARFSourceLanguage, SourceText,
    },
    llvm_sys::core::{LLVMContext, LLVMModule},
    to_llvm_ir,
};

/// Converts `module` to LLVM IR, with the debug data of `level`.
///
/// At `Full`, the `DIFile` of each file embeds its source text if `embed_source` is set. NVPTX
/// must not set it: `ptxas` rejects the `.file` directive with a source text.
///
/// The compile directory is the directory that has the relative source files on this computer,
/// if one does ([`kernel_source_root`]). At `Full`, the file must have the compiled text.
pub(crate) fn convert_module(
    ctx: &Context,
    llvm_ctx: &LLVMContext,
    module: ModuleOp,
    level: DebugInfo,
    embed_source: bool,
) -> pliron::result::Result<LLVMModule> {
    if level == DebugInfo::None {
        return to_llvm_ir::convert_module(ctx, llvm_ctx, module);
    }
    let mut options = DebugInfoOptions::default();
    options.emission_kind = match level {
        DebugInfo::Full => EmissionKind::Full,
        _ => EmissionKind::LineTablesOnly,
    };
    options.language = LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageRust;
    options.producer = "cubecl".to_string();
    options.optimized = true;
    let Some(debug) = ctx.try_aux_ty::<DebugState>() else {
        return to_llvm_ir::convert_module_with_debug_info(ctx, llvm_ctx, module, options);
    };
    // Only `Full` has the source text, as the macro records it only then.
    let md5s = source_md5s(debug, level);
    if let Some(root) = kernel_source_root(debug, &md5s) {
        options.directory = root;
    }
    if embed_source && !md5s.is_empty() {
        for (path, text) in debug.sources() {
            let md5 = md5s[path.as_str()].to_string();
            options
                .source_text
                .insert(path.clone(), SourceText { text, md5 });
        }
        // LLVM writes the source text into the object only with DWARF 5.
        options.dwarf_version = 5;
    }
    to_llvm_ir::convert_module_with_debug_info(ctx, llvm_ctx, module, options)
}

/// Runs LLVM's verifier on the debug data of `module`, in a debug build of cubecl only. The test
/// suites build with debug data, so they check every kernel.
///
/// # Panics
/// When the debug data does not verify: the conversion has a bug.
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
pub(crate) fn check_debug_info(module: &LlvmModule, kernel_name: &str, level: DebugInfo) {
    #[cfg(debug_assertions)]
    if level != DebugInfo::None
        && let Err(err) = module.verify()
    {
        panic!("the debug data of '{kernel_name}' does not verify: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cpu::jit::engine::to_llvm_module,
        shared::{
            PlironOptions,
            base::{LoweredCpu, lower_cpu},
            offline_kernels::{nested_calls_kernel, nested_calls_with_source_kernel},
        },
    };
    use cubecl_core::runtime_tests::offline::{SOURCE_PATH, source_line};
    use cubecl_runtime::{debug_source::md5_hex, kernel::CubeKernel};
    use pliron::{graph::walkers::uninterruptible::immutable::walk_op, location::Location};

    fn lowered(kernel: &impl CubeKernel) -> LoweredCpu {
        lower_cpu(kernel.define(), &PlironOptions::default()).unwrap()
    }

    /// The LLVM IR of `lowered`, with the debug data of `level`.
    fn ir_at(lowered: &LoweredCpu, level: DebugInfo) -> String {
        let module =
            to_llvm_module(&lowered.ctx, lowered.module, &lowered.kernel_name, level).unwrap();
        module.verify().unwrap();
        module.print()
    }

    /// The LLVM IR of `kernel` for the CPU.
    fn cpu_ir(kernel: &impl CubeKernel) -> String {
        let lowered = lowered(kernel);
        ir_at(&lowered, lowered.debug_info)
    }

    /// Each op of `lowered`.
    fn ops(lowered: &LoweredCpu) -> Vec<Ptr<Operation>> {
        let mut ops = Vec::new();
        walk_op(
            &lowered.ctx,
            &mut ops,
            &WALKCONFIG_PREORDER_FORWARD,
            lowered.module.get_operation(),
            |_ctx, ops, node| {
                if let IRNode::Operation(op) = node {
                    ops.push(op);
                }
            },
        );
        ops
    }

    /// The function name and the line of the innermost frame of `loc`.
    fn innermost(loc: &Location) -> Option<(&str, u32)> {
        match loc {
            Location::CallSite { callee, .. } => innermost(callee),
            Location::Named { name, child_loc } => match child_loc.as_ref() {
                Location::SrcPos { pos, .. } => Some((name, u32::try_from(pos.line).ok()?)),
                _ => None,
            },
            _ => None,
        }
    }

    /// The lowering passes keep the location of each op that comes from source: each float op,
    /// store and call. The entry ABI adds the loops over the units, which have no source and get
    /// line 0 in the DWARF. Each statement keeps its own line: `KeepLocation` gives a rewritten op
    /// the location of the op it replaces, where the fallback pass alone would give the location
    /// of the op before it.
    #[test]
    fn lowering_keeps_source_locations() {
        let lowered = lowered(&nested_calls_kernel(DebugInfo::LineTables));
        let ctx = &lowered.ctx;
        let unlocated = ops(&lowered)
            .into_iter()
            .map(|op| (Operation::get_opid(op, ctx).to_string(), op))
            .filter(|(name, op)| {
                let from_source = name == "llvm.store"
                    || name.starts_with("llvm.call")
                    || (name.starts_with("llvm.f")
                        && !["llvm.func", "llvm.fence"].contains(&name.as_str()));
                from_source && op.deref(ctx).loc().is_unknown()
            })
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert!(
            unlocated.is_empty(),
            "ops without a location: {unlocated:?}"
        );

        let lines = ops(&lowered)
            .into_iter()
            .filter_map(|op| {
                let loc = op.deref(ctx).loc();
                let (name, line) = innermost(&loc)?;
                Some((name.to_string(), line))
            })
            .collect::<Vec<_>>();
        for (function, text) in [
            ("square_third", "let y = x * x;"),
            ("square_third", "y / 3.0"),
            ("doubled", "third * 2.0"),
        ] {
            let line = source_line(text);
            assert!(
                lines.contains(&(function.to_string(), line)),
                "no op of `{text}` in {lines:?}"
            );
        }
    }

    /// The directory and the `DIFile` line of `file` in `ir`.
    fn difile<'a>(ir: &'a str, file: &str) -> (&'a str, &'a str) {
        let line = ir
            .lines()
            .find(|line| line.contains("DIFile(") && line.contains(file))
            .unwrap_or_else(|| panic!("no `DIFile` of {file}:\n{ir}"));
        let directory = line
            .split_once("directory: \"")
            .and_then(|(_, rest)| rest.split_once('"'))
            .map_or("", |(directory, _)| directory);
        (directory, line)
    }

    /// At `Full`, the `DIFile` of the kernel file embeds its text, in DWARF 5. Its MD5 is the MD5
    /// of the file on disk, so a debugger accepts the file as the source.
    #[test]
    fn full_debug_info_embeds_the_source_text() {
        let ir = cpu_ir(&nested_calls_with_source_kernel());
        let (directory, file) = difile(&ir, SOURCE_PATH);
        assert!(file.contains("source: \""), "{file}");
        let checksum = file
            .split_once("checksum: \"")
            .and_then(|(_, rest)| rest.split_once('"'))
            .map_or_else(|| panic!("no checksum: {file}"), |(checksum, _)| checksum);
        let on_disk = std::fs::read(std::path::Path::new(directory).join(SOURCE_PATH)).unwrap();
        assert_eq!(checksum, md5_hex(on_disk));
        assert!(ir.contains("!\"Dwarf Version\", i32 5"), "{ir}");
    }

    /// `LineTables` asks for line tables only, and without the text from the macro it embeds no
    /// source and keeps DWARF 4.
    #[test]
    fn line_tables_embed_no_source_text() {
        let ir = cpu_ir(&nested_calls_kernel(DebugInfo::LineTables));
        assert!(ir.contains("emissionKind: LineTablesOnly"), "{ir}");
        assert!(!ir.contains("source: \""), "{ir}");
        assert!(!ir.contains("checksum:"), "{ir}");
        assert!(ir.contains("!\"Dwarf Version\", i32 4"), "{ir}");
    }

    /// The tests run in the crate directory, and the workspace root, which `file!()` is relative
    /// to, is a parent. At both levels, the compile directory is that root.
    #[test]
    fn the_compile_directory_has_the_source_files() {
        for kernel_ir in [
            cpu_ir(&nested_calls_with_source_kernel()),
            cpu_ir(&nested_calls_kernel(DebugInfo::LineTables)),
        ] {
            let (directory, _) = difile(&kernel_ir, SOURCE_PATH);
            assert!(
                std::path::Path::new(directory).join(SOURCE_PATH).is_file(),
                "`{directory}` does not have {SOURCE_PATH}"
            );
        }
    }

    /// The ops have locations in a `dev` build. The level `None` must still give no debug data.
    #[test]
    fn no_debug_data_without_debug_info() {
        let ir = ir_at(
            &lowered(&nested_calls_kernel(DebugInfo::LineTables)),
            DebugInfo::None,
        );
        assert!(
            !ir.contains("!dbg") && !ir.contains("DICompileUnit"),
            "{ir}"
        );
    }
}
