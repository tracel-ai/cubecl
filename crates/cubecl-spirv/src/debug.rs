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

use cubecl_core::ir::{self as core, KernelDefinition};
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

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugFunction {
    id: Word,
    name: Word,
    source: Word,
    line_offset: u32,
}

#[derive(Clone, Debug, Default)]
pub struct FunctionCall {
    id: Word,
    name: String,
    inlined_at: Option<Word>,
}

#[derive(Clone, Debug)]
pub struct DebugInfo {
    pub source: Word,
    pub name: Word,
    pub name_str: String,
    pub compilation_unit: Word,
    pub entry_point: Word,
    pub functions: Vec<FunctionCall>,
    strings: HashMap<String, Word>,
    sources: HashMap<String, Word>,
    function_defs: HashMap<String, DebugFunction>,
    inlined_at_defs: Vec<(Word, u32, Option<Word>, Word)>,
    function_ty: Word,

    line: Vec<u32>,
    col: Vec<u32>,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_debug(&mut self, kernel: KernelDefinition) {
        if self.debug {
            let mut strings = HashMap::new();
            let name = self.string(&kernel.kernel_name);
            strings.insert(kernel.kernel_name.clone(), name);

            let source = self.id();

            let compilation_unit = self.id();
            let flags = self.const_u32(DebugInfoFlags::None.0);
            let return_ty = self.type_void();
            let function_ty =
                self.void_debug(None, Instructions::DebugTypeFunction, [flags, return_ty]);
            let main_id = self.id();

            let main = DebugFunction {
                id: main_id,
                name,
                source,
                ..Default::default()
            };

            let mut function_defs = HashMap::default();
            function_defs.insert(kernel.kernel_name.clone(), main);

            let main_call = FunctionCall {
                id: main_id,
                name: kernel.kernel_name.clone(),
                inlined_at: None,
            };

            self.debug_info = Some(DebugInfo {
                source,
                name,
                name_str: kernel.kernel_name,
                compilation_unit,
                entry_point: 0,
                functions: vec![main_call],
                strings,
                inlined_at_defs: Default::default(),
                sources: Default::default(),
                function_defs,
                function_ty,

                line: vec![0],
                col: vec![0],
            });
        }
    }

    fn debug_string(&mut self, value: impl Into<String>) -> Word {
        let value: String = value.into();
        if let Some(id) = self.debug_info().strings.get(&value).copied() {
            id
        } else {
            let id = self.string(value.clone());
            self.debug_info().strings.insert(value, id);
            id
        }
    }

    pub fn declare_main(&mut self, kernel_name: &str) -> (Word, impl Fn(&mut Self)) {
        let void = self.type_void();
        let voidf = self.type_function(void, vec![]);

        let debug_function = self.debug.then(|| {
            let debug_function = self.debug_function(kernel_name);
            self.debug_info().entry_point = debug_function;
            self.debug_info().functions.push(FunctionCall {
                id: debug_function,
                name: kernel_name.to_string(),
                ..Default::default()
            });
            debug_function
        });

        let main = self
            .begin_function(void, None, FunctionControl::NONE, voidf)
            .unwrap();
        self.debug_name(main, kernel_name);

        let setup = move |b: &mut Self| {
            if let Some(debug_function) = debug_function {
                b.void_op(Instructions::DebugScope, [debug_function]);
                b.void_op(
                    Instructions::DebugFunctionDefinition,
                    [debug_function, main],
                );
            }
        };

        (main, setup)
    }

    pub fn compile_debug(&mut self, debug: core::DebugInfo) {
        if self.debug {
            match debug {
                core::DebugInfo::Source {
                    file_name,
                    source,
                    line_offset,
                } => {
                    let name = self.debug_string(file_name.clone());
                    let source = self.debug_source(file_name.clone(), source);

                    let current_func = self.debug_info().functions.last().unwrap().name.clone();
                    let debug_func = self.debug_info().function_defs.get_mut(&current_func);

                    if let Some(debug_func) = debug_func {
                        debug_func.name = name;
                        debug_func.source = source;
                        debug_func.line_offset = line_offset;
                    }

                    self.line(name, 0, 0);
                }
                core::DebugInfo::BeginCall { name, line, col } => {
                    let func = self.debug_function(name.clone());
                    let parent = self.debug_info().functions.last().unwrap().clone();

                    let parent_offset = self.debug_info().function_defs[&parent.name].line_offset;

                    let line_rel = line - parent_offset;

                    *self.debug_info().line.last_mut().unwrap() = line_rel;
                    *self.debug_info().col.last_mut().unwrap() = col;

                    self.debug_info().line.push(0);
                    self.debug_info().col.push(0);

                    let inlined_at = self.inlined_at(parent.id, line_rel, parent.inlined_at);

                    if self.is_last_instruction_scope() {
                        let _ = self.pop_instruction();
                    }

                    self.debug_info().functions.push(FunctionCall {
                        id: func,
                        name,
                        inlined_at: Some(inlined_at),
                    });
                    self.void_op(Instructions::DebugScope, [func, inlined_at]);
                }
                core::DebugInfo::EndCall => {
                    self.debug_info().functions.pop();
                    self.debug_info().line.pop();
                    self.debug_info().col.pop();

                    if self.is_last_instruction_scope() {
                        let _ = self.pop_instruction();
                    }
                    let func = self.debug_info().functions.last().unwrap().clone();
                    let debug_func = self.debug_info().function_defs[&func.name];
                    self.void_op(Instructions::DebugScope, [func.id]);

                    let line = *self.debug_info().line.last().unwrap();
                    let col = *self.debug_info().col.last().unwrap();

                    self.line(debug_func.name, line, col);
                }
                core::DebugInfo::Print {
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
            };
        }
    }

    // Deduplicated inlined_at
    fn inlined_at(&mut self, parent: Word, line: u32, inline_parent: Option<Word>) -> Word {
        let existing = self
            .debug_info()
            .inlined_at_defs
            .iter()
            .find(|it| it.0 == parent && it.1 == line && it.2 == inline_parent);
        if let Some(existing) = existing {
            existing.3
        } else {
            let id = self.id();
            self.debug_info()
                .inlined_at_defs
                .push((parent, line, inline_parent, id));
            id
        }
    }

    // Deduplicated debug_source
    fn debug_source(&mut self, file: String, source: String) -> Word {
        let existing = self.debug_info().sources.get(&file);
        if let Some(existing) = existing {
            *existing
        } else {
            let file_id = self.debug_string(&file);
            let source_id = self.debug_string(&source);
            self.source(SourceLanguage::Unknown, 1, Some(file_id), Some(source));
            let source = self.void_debug(None, Instructions::DebugSource, [file_id, source_id]);
            self.debug_info().sources.insert(file, source);
            source
        }
    }

    fn debug_function(&mut self, name: impl Into<String>) -> Word {
        let name: String = name.into();
        if let Some(func) = self.debug_info().function_defs.get(&name).copied() {
            func.id
        } else {
            let name_id = self.debug_string(&name);
            let id = self.id();
            let source = self.debug_info().source;

            self.debug_info().function_defs.insert(
                name,
                DebugFunction {
                    id,
                    name: name_id,
                    source,
                    ..Default::default()
                },
            );
            id
        }
    }

    pub fn finish_debug(&mut self) {
        let compilation_unit = self.debug_info().compilation_unit;
        let entry_name = self.debug_info().name_str.clone();
        let source = self.debug_info().function_defs[&entry_name].source;
        let version = self.const_u32(1);
        let dwarf = self.const_u32(4);
        let lang = self.const_u32(SourceLanguage::Unknown as u32);
        self.void_debug(
            Some(compilation_unit),
            Instructions::DebugCompilationUnit,
            [version, dwarf, source, lang],
        );

        if let Some(debug_info) = self.debug_info.clone() {
            for function in debug_info.function_defs.values() {
                self.declare_debug_function(function);
            }

            for (parent, line, inline_parent, id) in debug_info.inlined_at_defs.iter().copied() {
                self.declare_inlined_at(id, parent, line, inline_parent);
            }
        }
    }

    fn declare_debug_function(&mut self, function: &DebugFunction) {
        let name_id = function.name;
        let flags = self.const_u32(DebugInfoFlags::None.0);
        let debug_type = self.debug_info().function_ty;
        let source = function.source;
        let compilation_unit = self.debug_info().compilation_unit;
        let zero = self.const_u32(0);
        self.void_debug(
            Some(function.id),
            Instructions::DebugFunction,
            [
                name_id,
                debug_type,
                source,
                zero,
                zero,
                compilation_unit,
                name_id,
                flags,
                zero,
            ],
        );
    }

    // Deduplicated inlined_at
    fn declare_inlined_at(
        &mut self,
        out_id: Word,
        parent: Word,
        line: u32,
        inline_parent: Option<Word>,
    ) {
        let line = self.const_u32(line);
        if let Some(inline_parent) = inline_parent {
            self.void_debug(
                Some(out_id),
                Instructions::DebugInlinedAt,
                [line, parent, inline_parent],
            )
        } else {
            self.void_debug(Some(out_id), Instructions::DebugInlinedAt, [line, parent])
        };
    }

    pub fn debug_scope(&mut self) {
        if self.debug {
            let func = self.debug_info().functions.last().unwrap().clone();
            let func = self.debug_info().function_defs[&func.name];
            let line = *self.debug_info().line.last().unwrap();
            let col = *self.debug_info().col.last().unwrap();
            self.line(func.name, line, col);
            self.void_op(Instructions::DebugScope, [func.id]);
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

    pub fn debug_info(&mut self) -> &mut DebugInfo {
        self.debug_info.as_mut().unwrap()
    }

    fn is_last_instruction_scope(&mut self) -> bool {
        let (selected_function, selected_block) =
            match (self.selected_function(), self.selected_block()) {
                (Some(f), Some(b)) => (f, b),
                _ => panic!("Detached instructions"),
            };

        let block = &self.module_ref().functions[selected_function].blocks[selected_block];

        let inst = block.instructions.last();
        let ext = self.state.extensions[DEBUG_EXT_NAME];

        if let Some(Instruction {
            class, operands, ..
        }) = inst
        {
            class.opcode == Op::ExtInst
                && operands.first() == Some(&Operand::IdRef(ext))
                && operands.get(1)
                    == Some(&Operand::LiteralExtInstInteger(
                        Instructions::DebugScope as u32,
                    ))
        } else {
            false
        }
    }
}
