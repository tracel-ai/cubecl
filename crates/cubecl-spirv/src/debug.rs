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

use std::iter;

use cubecl_core::ir::{self as core, CubeSource, SourceLoc};
use hashbrown::HashMap;
use rspirv::{
    dr::{Instruction, Operand},
    spirv::{FunctionControl, Op, SourceLanguage, Word},
};

use crate::{
    extensions::NonSemanticShaderDebugInfo100::{DebugInfoFlags, Instructions},
    SpirvCompiler, SpirvTarget,
};

pub const DEBUG_EXT_NAME: &str = "NonSemantic.Shader.DebugInfo.100";
pub const PRINT_EXT_NAME: &str = "NonSemantic.DebugPrintf";
pub const SIGNATURE: &str = concat!(env!("CARGO_PKG_NAME"), " v", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Copy, Debug, Default)]
pub struct FunctionDefinition {
    id: Word,
    name: Word,
    file_name: Word,
    source: SourceFile,
    line: u32,
    col: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct InlinedAt {
    id: Word,
    func: Word,
    line: u32,
    parent: Option<Word>,
}

#[derive(Clone, Debug, Default)]
pub struct FunctionCall {
    definition: FunctionDefinition,
    scope: Word,
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
    functions: HashMap<CubeSource, FunctionDefinition>,
    /// InlinedAt definitions
    inlined_at: HashMap<(Word, u32, Option<Word>), Word>,
    /// Debug strings
    strings: HashMap<String, Word>,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_debug(&mut self) {
        if self.debug_symbols {
            let flags = self.const_u32(DebugInfoFlags::None.0);
            let return_ty = self.type_void();
            let function_ty =
                self.void_debug(None, Instructions::DebugTypeFunction, [flags, return_ty]);
            let entry_loc = self.opt.root_scope.entry_loc.clone().unwrap();

            self.debug_info = Some(DebugInfo {
                function_ty,
                stack: Default::default(),
                definitions: Default::default(),
                previous_loc: Some(entry_loc.clone()),
            });

            self.collect_sources();

            let entry_def = self.definitions().functions[&entry_loc.source];
            let args = self.debug_string("");
            let signature = self.debug_string(SIGNATURE);
            self.void_debug(
                None,
                Instructions::DebugEntryPoint,
                [
                    entry_def.id,
                    entry_def.source.compilation_unit,
                    signature,
                    args,
                ],
            );
        }
    }

    /// Collect sources ahead of time so line numbers and source file names are correct
    fn collect_sources(&mut self) {
        let cube_fns = self.opt.root_scope.sources.borrow().clone();
        let mut sources = HashMap::<_, String>::new();
        for cube_fn in cube_fns.iter() {
            let source = sources.entry(cube_fn.file.clone()).or_default();
            let fn_lines = cube_fn.source_text.lines().count();
            let last_line = cube_fn.line as usize + fn_lines;
            let mut lines = source.lines().map(|it| it.to_string()).collect::<Vec<_>>();
            if lines.len() < last_line {
                lines.extend(iter::repeat(String::new()).take(last_line - lines.len()));
            }
            for (line, fn_text) in lines
                .iter_mut()
                .skip(cube_fn.line as usize)
                .take(fn_lines)
                .zip(cube_fn.source_text.lines())
            {
                *line = fn_text.to_string();
            }
            *source = lines.join("\n");
        }

        for (file, source_text) in sources {
            self.debug_source(file, source_text);
        }

        for cube_fn in cube_fns.into_iter() {
            let source = self.definitions().source_files[cube_fn.file.as_ref()];
            let function = FunctionDefinition {
                id: self.id(),
                name: self.debug_string(cube_fn.function_name.as_ref()),
                file_name: self.debug_string(cube_fn.file.as_ref()),
                source,
                line: cube_fn.line,
                col: cube_fn.column,
            };
            self.declare_debug_function(&function);
            self.definitions().functions.insert(cube_fn, function);
        }
    }

    fn debug_string(&mut self, value: impl Into<String>) -> Word {
        let value: String = value.into();
        if let Some(id) = self.debug_info().definitions.strings.get(&value).copied() {
            id
        } else {
            let id = self.string(value.clone());
            self.debug_info().definitions.strings.insert(value, id);
            id
        }
    }

    pub fn declare_main(&mut self, kernel_name: &str) -> (Word, impl Fn(&mut Self)) {
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
                scope: 0,
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
                let scope = b.void_op(Instructions::DebugScope, [func_id]);
                b.stack_top().scope = scope;
                b.void_op(Instructions::DebugFunctionDefinition, [func_id, main]);
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
                        let file_name = self.debug_string(loc.source.file.as_ref());
                        self.line(file_name, loc.line, loc.column);
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
            let parent = self.stack_top().inlined_at;
            self.inlined_at(call_fn.id, prev.line, parent)
        });
        self.stack_top().definition = call_fn;
        self.stack_top().inlined_at = inlined_at;
    }

    pub fn compile_debug(&mut self, debug: core::NonSemantic) {
        if self.debug_symbols {
            match debug {
                core::NonSemantic::Print {
                    format_string,
                    args,
                } => {
                    let ext = self.state.extensions[PRINT_EXT_NAME];
                    let void = self.type_void();
                    let args = args
                        .into_iter()
                        .map(|arg| {
                            let var = self.compile_variable(arg);
                            Operand::IdRef(self.read(&var))
                        })
                        .collect::<Vec<_>>();
                    let mut operands = vec![Operand::IdRef(self.debug_string(format_string))];
                    operands.extend(args);
                    self.ext_inst(void, None, ext, 1, operands).unwrap();
                }
                core::NonSemantic::Comment { .. } => {
                    // Comments not supported for SPIR-V
                }
                core::NonSemantic::DebugScopeStart => {
                    let new_top = self.stack_top().clone();
                    self.stack().push(new_top);
                }
                core::NonSemantic::DebugScopeEnd => {
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

    // Deduplicated inlined_at
    fn inlined_at(&mut self, func: Word, line: u32, parent: Option<Word>) -> Word {
        let existing = self.definitions().inlined_at.get(&(func, line, parent));
        if let Some(existing) = existing {
            *existing
        } else {
            let id = self.id();
            let inlined_at = InlinedAt {
                id,
                func,
                line,
                parent,
            };
            self.declare_inlined_at(&inlined_at);
            self.definitions()
                .inlined_at
                .insert((func, line, parent), id);
            id
        }
    }

    // Deduplicated debug_source
    fn debug_source(&mut self, file: impl AsRef<str>, source_text: impl AsRef<str>) -> SourceFile {
        let existing = self.definitions().source_files.get(file.as_ref());
        if let Some(existing) = existing {
            *existing
        } else {
            let file_id = self.debug_string(file.as_ref());
            let source_id = self.debug_string(source_text.as_ref());

            self.source(
                SourceLanguage::Unknown,
                1,
                Some(file_id),
                Some(source_text.as_ref()),
            );

            let source = self.void_debug(None, Instructions::DebugSource, [file_id, source_id]);
            let comp_unit = self.debug_compilation_unit(source);
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

    fn debug_compilation_unit(&mut self, source: Word) -> Word {
        let version = self.const_u32(1);
        let dwarf = self.const_u32(4);
        let language = self.const_u32(SourceLanguage::Unknown as u32);
        self.void_debug(
            None,
            Instructions::DebugCompilationUnit,
            [version, dwarf, source, language],
        )
    }

    fn declare_debug_function(&mut self, function: &FunctionDefinition) {
        let name_id = function.name;
        let flags = self.const_u32(DebugInfoFlags::None.0);
        let debug_type = self.debug_info().function_ty;
        let source = function.source.id;
        let compilation_unit = function.source.compilation_unit;
        let line = self.const_u32(function.line);
        let col = self.const_u32(function.col);
        let zero = self.const_u32(0);
        self.void_debug(
            Some(function.id),
            Instructions::DebugFunction,
            [
                name_id,
                debug_type,
                source,
                line,
                col,
                compilation_unit,
                name_id,
                flags,
                zero,
            ],
        );
    }

    // Deduplicated inlined_at
    fn declare_inlined_at(&mut self, inlined_at: &InlinedAt) {
        let line = self.const_u32(inlined_at.line);
        if let Some(inline_parent) = inlined_at.parent {
            self.void_debug(
                Some(inlined_at.id),
                Instructions::DebugInlinedAt,
                [line, inlined_at.func, inline_parent],
            )
        } else {
            self.void_debug(
                Some(inlined_at.id),
                Instructions::DebugInlinedAt,
                [line, inlined_at.func],
            )
        };
    }

    pub fn debug_start_block(&mut self) {
        if self.debug_symbols {
            let loc = self.debug_info().previous_loc.clone().unwrap();
            let func = self.stack_top().definition;
            let inlined = self.stack_top().inlined_at;
            let line = loc.line;
            let col = loc.column;

            self.line(func.file_name, line, col);
            let scope = self.void_op(Instructions::DebugScope, [func.id]);
            self.stack_top().scope = scope;
        }
    }

    fn void_debug<const N: usize>(
        &mut self,
        out_id: Option<Word>,
        instruction: Instructions,
        operands: [Word; N],
    ) -> Word {
        let ext = self.state.extensions[DEBUG_EXT_NAME];
        let void = self.type_void();
        let operands = operands.into_iter().map(Operand::IdRef).collect::<Vec<_>>();
        let out = out_id.unwrap_or_else(|| self.id());
        let mut ops = vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(instruction as u32),
        ];
        ops.extend(operands);
        let inst = Instruction::new(Op::ExtInst, Some(void), Some(out), ops);
        self.module_mut().types_global_values.push(inst);
        out
    }

    fn void_op<const N: usize>(&mut self, instruction: Instructions, operands: [Word; N]) -> Word {
        let ext = self.state.extensions[DEBUG_EXT_NAME];
        let void = self.type_void();
        let operands = operands.into_iter().map(Operand::IdRef).collect::<Vec<_>>();
        self.ext_inst(void, None, ext, instruction as u32, operands)
            .unwrap()
    }

    #[track_caller]
    pub fn debug_info(&mut self) -> &mut DebugInfo {
        self.debug_info.as_mut().unwrap()
    }
}
