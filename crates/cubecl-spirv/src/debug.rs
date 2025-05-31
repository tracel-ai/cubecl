//! Uses non-semantic extensions to add debug info to the generated SPIR-V.
//!
//! Adds a dummy source with the kernel name as the file name and a dummy compilation unit.
//! Then marks the top level function, and any inlined functions marked by debug instructions with
//! their corresponding `OpDebugFunction`s and `OpDebugInlinedAt` instructions, and marks the extent
//! with `OpDebugScope`.
//!
//! Also adds dummy `OpLine` instructions to allow Nsight to see the file name. Might add real
//! line numbers in the future if possible.
//!
//! To get proper debugging, every instruction must be inside an `OpDebugScope` referencing the
//! appropriate function.
//!
//! # Deduplication
//!
//! All debug instructions are deduplicated to ensure minimal binary size when functions are called
//! in a loop or other similar situations.

use std::borrow::Cow;

use cubecl_core::ir::{self as core, CubeFnSource, SourceLoc, Variable};
use hashbrown::HashMap;
use rspirv::spirv::{FunctionControl, Word};
use rspirv_ext::{
    spirv::DebugInfoFlags,
    sr::{
        nonsemantic_debugprintf::DebugPrintfBuilder,
        nonsemantic_shader_debuginfo_100::DebugInfoBuilder,
    },
};

use crate::{SpirvCompiler, SpirvTarget};

pub const SIGNATURE: &str = concat!(env!("CARGO_PKG_NAME"), " v", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Copy, Debug, Default)]
pub struct FunctionDefinition {
    id: Word,
    source: SourceFile,
    line: u32,
    col: u32,
}

#[derive(Clone, Debug, Default)]
pub struct FunctionCall {
    definition: FunctionDefinition,
    inlined_at: Option<Word>,
}

#[derive(Clone, Debug)]
pub struct DebugInfo {
    function_ty: Word,

    stack: Vec<FunctionCall>,
    definitions: Definitions,
    previous_loc: Option<SourceLoc>,
}

#[derive(Clone, Copy, Debug, Default)]
struct SourceFile {
    /// Id of the `DebugSource` instruction
    id: Word,
    /// Id of the compilation unit for this file
    compilation_unit: Word,
}

#[derive(Clone, Debug, Default)]
struct Definitions {
    /// source files
    source_files: HashMap<String, SourceFile>,
    /// map of call names to definitions
    functions: HashMap<CubeFnSource, FunctionDefinition>,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_debug(&mut self) {
        if self.debug_enabled() {
            let return_ty = self.type_void();
            let function_ty = self.debug_type_function(DebugInfoFlags::NONE, return_ty, []);
            let entry_loc = self.opt.root_scope.debug.entry_loc.clone().unwrap();

            self.debug_info = Some(DebugInfo {
                function_ty,
                stack: Default::default(),
                definitions: Default::default(),
                previous_loc: Some(entry_loc.clone()),
            });

            self.collect_sources();

            let entry_def = self.definitions().functions[&entry_loc.source];
            self.debug_entry_point(
                entry_def.id,
                entry_def.source.compilation_unit,
                SIGNATURE,
                "",
            );
        } else if self.debug_symbols {
            let return_ty = self.type_void();
            let function_ty = self.debug_type_function(DebugInfoFlags::NONE, return_ty, []);
            self.debug_info = Some(DebugInfo {
                function_ty,
                stack: Default::default(),
                definitions: Default::default(),
                previous_loc: None,
            });
        }
    }

    /// Collect sources ahead of time so line numbers and source file names are correct
    fn collect_sources(&mut self) {
        let cube_fns = self.opt.root_scope.debug.sources.borrow().clone();
        let mut sources = HashMap::new();
        for cube_fn in cube_fns.iter() {
            // If source is missing, don't override since it might exist from another function in the
            // same file. If it's not empty, just override since they're identical.
            if cube_fn.source_text.is_empty() {
                sources
                    .entry(cube_fn.file.clone())
                    .or_insert(cube_fn.source_text.clone());
            } else {
                sources.insert(cube_fn.file.clone(), cube_fn.source_text.clone());
            }
        }

        for (file, source_text) in sources {
            self.debug_source_dedup(file, source_text);
        }

        for cube_fn in cube_fns.into_iter() {
            let source = self.definitions().source_files[cube_fn.file.as_ref()];
            let name = cube_fn.function_name.as_ref();
            let mut function = FunctionDefinition {
                id: 0,
                source,
                line: cube_fn.line,
                col: cube_fn.column,
            };
            self.declare_debug_function(name, &mut function);
            self.definitions().functions.insert(cube_fn, function);
        }
    }

    pub fn declare_main(&mut self, kernel_name: &str) -> (Word, impl Fn(&mut Self) + 'static) {
        let void = self.type_void();
        let voidf = self.type_function(void, vec![]);

        let definition = self
            .debug_info
            .as_ref()
            .and_then(|info| info.previous_loc.clone())
            .map(|loc| self.definitions().functions[&loc.source]);

        if let Some(definition) = definition {
            let main_call = FunctionCall {
                definition,
                inlined_at: None,
            };

            self.stack().push(main_call);
        }

        let main = self
            .begin_function(void, None, FunctionControl::NONE, voidf)
            .unwrap();
        self.debug_name(main, kernel_name);

        let func_id = definition.map(|it| it.id);

        let setup = move |b: &mut Self| {
            if let Some(func_id) = func_id {
                b.debug_start_block();
                b.debug_function_definition(func_id, main).unwrap();
            }
        };

        (main, setup)
    }

    pub fn set_source_loc(&mut self, loc: &Option<SourceLoc>) {
        if let Some(loc) = loc {
            match self.debug_info().previous_loc.clone() {
                Some(prev) if &prev != loc => {
                    self.debug_info().previous_loc = Some(loc.clone());
                    if prev.source != loc.source {
                        self.update_call(loc.clone(), prev);
                        self.debug_start_block();
                    } else {
                        let source = self.stack_top().definition.source.id;
                        self.debug_line(source, loc.line, loc.line, loc.column, loc.column)
                            .unwrap();
                    }
                }
                _ => {}
            }
        }
    }

    fn update_call(&mut self, loc: SourceLoc, prev: SourceLoc) {
        let is_inlined = self.stack().len() > 1;
        let call_fn = self.definitions().functions[&loc.source];
        let inlined_at = is_inlined.then(|| {
            let parent = self.stack_prev().clone();
            self.debug_inlined_at(parent.definition.id, prev.line, parent.inlined_at)
        });
        self.stack_top().definition = call_fn;
        self.stack_top().inlined_at = inlined_at;
    }

    pub fn compile_debug(&mut self, debug: core::NonSemantic) {
        if self.debug_enabled() {
            match debug {
                core::NonSemantic::Print {
                    format_string,
                    args,
                } => {
                    let args = args
                        .into_iter()
                        .map(|arg| {
                            let var = self.compile_variable(arg);
                            self.read(&var)
                        })
                        .collect::<Vec<_>>();
                    self.debug_printf(format_string, args).unwrap();
                }
                core::NonSemantic::Comment { .. } => {
                    // Comments not supported for SPIR-V
                }
                core::NonSemantic::EnterDebugScope => {
                    let new_top = self.stack_top().clone();
                    self.stack().push(new_top);
                }
                core::NonSemantic::ExitDebugScope => {
                    self.stack().pop();
                }
            };
        }
    }

    fn definitions(&mut self) -> &mut Definitions {
        &mut self.debug_info().definitions
    }

    fn stack(&mut self) -> &mut Vec<FunctionCall> {
        &mut self.debug_info().stack
    }

    fn stack_top(&mut self) -> &mut FunctionCall {
        self.debug_info().stack.last_mut().unwrap()
    }

    fn stack_prev(&mut self) -> &mut FunctionCall {
        self.debug_info().stack.iter_mut().nth_back(1).unwrap()
    }

    // Deduplicated debug_source
    fn debug_source_dedup(
        &mut self,
        file: impl AsRef<str>,
        source_text: impl AsRef<str>,
    ) -> SourceFile {
        let existing = self.definitions().source_files.get(file.as_ref());
        if let Some(existing) = existing {
            *existing
        } else {
            let source = self.debug_source(file.as_ref(), Some(source_text.as_ref()));

            let comp_unit = {
                use rspirv_ext::dr::autogen_nonsemantic_shader_debuginfo_100::DebugInfoOpBuilder;

                let version = self.const_u32(1);
                let dwarf_version = self.const_u32(5);
                let language = self.const_u32(13);
                self.debug_compilation_unit_id(None, version, dwarf_version, source, language)
            };

            let source_file = SourceFile {
                id: source,
                compilation_unit: comp_unit,
            };
            self.definitions()
                .source_files
                .insert(file.as_ref().into(), source_file);
            source_file
        }
    }

    fn declare_debug_function(&mut self, name: &str, function: &mut FunctionDefinition) {
        let debug_type = self.debug_info().function_ty;

        function.id = self.debug_function(
            name,
            debug_type,
            function.source.id,
            function.line,
            function.col,
            function.source.compilation_unit,
            name,
            DebugInfoFlags::NONE,
            function.line,
            None,
        );
    }

    pub fn debug_start_block(&mut self) {
        if self.debug_enabled() {
            let loc = self.debug_info().previous_loc.clone().unwrap();
            let func = self.stack_top().definition;
            let inlined = self.stack_top().inlined_at;

            if let Some(inlined) = inlined {
                self.debug_scope(func.id, Some(inlined)).unwrap()
            } else {
                self.debug_scope(func.id, None).unwrap()
            };
            self.debug_line(func.source.id, loc.line, loc.line, loc.column, loc.column)
                .unwrap();
        }
    }

    fn debug_enabled(&self) -> bool {
        self.debug_symbols && self.opt.root_scope.debug.entry_loc.is_some()
    }

    #[track_caller]
    pub fn debug_info(&mut self) -> &mut DebugInfo {
        self.debug_info.as_mut().unwrap()
    }

    pub fn debug_name(&mut self, var: Word, name: impl Into<String>) {
        if self.debug_symbols {
            self.name(var, name);
        }
    }

    pub fn debug_var_name(&mut self, id: Word, var: Variable) {
        if self.debug_symbols {
            let name = self.name_of_var(var);
            self.debug_name(id, name);
        }
    }

    pub fn name_of_var(&mut self, var: Variable) -> Cow<'static, str> {
        let var_names = self.opt.root_scope.debug.variable_names.clone();
        let debug_name = var_names.borrow().get(&var).cloned();
        debug_name.unwrap_or_else(|| var.to_string().into())
    }
}
