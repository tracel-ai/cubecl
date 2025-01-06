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

use cubecl_core::ir::{self as core, Operation};
use cubecl_opt::Optimizer;
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

impl InlinedAt {
    pub fn matches(&self, func: Word, line: u32, parent: Option<Word>) -> bool {
        self.func == func && self.line == line && self.parent == parent
    }
}

#[derive(Clone, Debug, Default)]
pub struct FunctionCall {
    id: Word,
    name: String,
    inlined_at: Option<Word>,
}

#[derive(Clone, Debug)]
pub struct DebugInfo {
    pub name: Word,
    pub name_str: String,
    function_ty: Word,

    stack: Vec<StackFrame>,
    definitions: Definitions,
}

#[derive(Clone, Debug)]
struct StackFrame {
    call: FunctionCall,
    line: u32,
    col: u32,
}

impl StackFrame {
    pub fn new(call: FunctionCall) -> Self {
        Self {
            call,
            line: 0,
            col: 0,
        }
    }
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
    functions: HashMap<String, FunctionDefinition>,
    /// InlinedAt definitions
    inlined_at: Vec<InlinedAt>,
    /// Debug strings
    strings: HashMap<String, Word>,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_debug(&mut self, kernel_name: String, opt: &Optimizer) {
        if self.debug {
            let name = self.string(&kernel_name);

            let flags = self.const_u32(DebugInfoFlags::None.0);
            let return_ty = self.type_void();
            let function_ty =
                self.void_debug(None, Instructions::DebugTypeFunction, [flags, return_ty]);

            self.debug_info = Some(DebugInfo {
                name,
                name_str: kernel_name.clone(),
                function_ty,
                stack: Default::default(),
                definitions: Default::default(),
            });

            self.definitions().strings.insert(kernel_name.clone(), name);
            self.collect_sources(kernel_name, opt);
        }
    }

    /// Collect sources ahead of time so line numbers and source file names are correct
    fn collect_sources(&mut self, kernel_name: String, opt: &Optimizer) {
        let name_id = self.debug_string(kernel_name.clone());
        let main_id = self.id();
        let main_source = SourceFile {
            id: self.id(),
            compilation_unit: self.id(),
        };
        self.definitions().functions.insert(
            kernel_name.clone(),
            FunctionDefinition {
                id: main_id,
                name: name_id,
                file_name: name_id,
                source: main_source,
                line: 0,
                col: 0,
            },
        );

        let mut calls = vec![kernel_name];

        for block_id in opt.node_ids() {
            for inst in opt.block(block_id).ops.borrow().values() {
                match &inst.operation {
                    Operation::Debug(core::DebugInfo::BeginCall { name, .. }) => {
                        calls.push(name.clone());
                        self.debug_function(name);
                    }
                    Operation::Debug(core::DebugInfo::EndCall) => {
                        calls.pop();
                    }
                    Operation::Debug(core::DebugInfo::Source {
                        name,
                        file_name,
                        line,
                        col,
                    }) => {
                        let call_name = calls.last().unwrap();

                        let name = self.debug_string(name);
                        let source = self.debug_source(file_name.clone());
                        let file_name = self.debug_string(file_name);

                        let debug_func = self.definitions().functions.get_mut(call_name).unwrap();

                        debug_func.name = name;
                        debug_func.file_name = file_name;
                        debug_func.source = source;
                        debug_func.line = *line;
                        debug_func.col = *col;
                    }
                    _ => {}
                }
            }
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
            .debug
            .then(|| self.definitions().functions[kernel_name]);

        if let Some(definition) = definition {
            let main_call = FunctionCall {
                id: definition.id,
                name: kernel_name.into(),
                inlined_at: None,
            };

            let root_frame = StackFrame {
                call: main_call,
                line: definition.line,
                col: definition.col,
            };

            self.stack().push(root_frame);
        }

        let main = self
            .begin_function(void, None, FunctionControl::NONE, voidf)
            .unwrap();
        self.debug_name(main, kernel_name);

        let func_id = definition.map(|it| it.id);

        let setup = move |b: &mut Self| {
            if let Some(func_id) = func_id {
                b.void_op(Instructions::DebugScope, [func_id]);
                b.void_op(Instructions::DebugFunctionDefinition, [func_id, main]);
            }
        };

        (main, setup)
    }

    pub fn compile_debug(&mut self, debug: core::DebugInfo) {
        if self.debug {
            match debug {
                core::DebugInfo::Source {
                    name,
                    file_name,
                    line,
                    col,
                } => {
                    let name = self.debug_string(name);
                    let source = self.debug_source(file_name.clone());
                    let file_name = self.debug_string(file_name);

                    let debug_func = self.current_function_def();

                    debug_func.name = name;
                    debug_func.file_name = file_name;
                    debug_func.source = source;
                    debug_func.line = line;
                    debug_func.col = col;

                    self.stack_top().line = line;
                    self.stack_top().col = col;
                }
                core::DebugInfo::BeginCall { name, line, col } => {
                    let func = self.debug_function(name.clone());

                    self.stack_top().line = line;
                    self.stack_top().col = col;

                    let parent = self.stack_top().call.clone();
                    let inlined_at = self.inlined_at(parent.id, line, parent.inlined_at);

                    if self.is_last_instruction_scope() {
                        let _ = self.pop_instruction();
                    }

                    let call = FunctionCall {
                        id: func,
                        name,
                        inlined_at: Some(inlined_at),
                    };
                    self.stack().push(StackFrame::new(call));

                    self.void_op(Instructions::DebugScope, [func, inlined_at]);
                }
                core::DebugInfo::EndCall => {
                    self.stack().pop();

                    if self.is_last_instruction_scope() {
                        let _ = self.pop_instruction();
                    }

                    let new_func = *self.current_function_def();
                    self.void_op(Instructions::DebugScope, [new_func.id]);

                    let line = self.stack_top().line;
                    let col = self.stack_top().col;

                    self.line(new_func.file_name, line, col);
                }
                core::DebugInfo::Line { line, col } => {
                    self.stack_top().line = line;
                    self.stack_top().col = col;

                    let file = self.current_function_def().file_name;

                    self.line(file, line, col);
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

    fn current_function_def(&mut self) -> &mut FunctionDefinition {
        let current_call_name = self.stack_top().call.name.clone();
        self.debug_info()
            .definitions
            .functions
            .get_mut(&current_call_name)
            .unwrap()
    }

    fn definitions(&mut self) -> &mut Definitions {
        &mut self.debug_info().definitions
    }

    fn stack(&mut self) -> &mut Vec<StackFrame> {
        &mut self.debug_info().stack
    }

    fn stack_top(&mut self) -> &mut StackFrame {
        self.debug_info().stack.last_mut().unwrap()
    }

    // Deduplicated inlined_at
    fn inlined_at(&mut self, func: Word, line: u32, parent: Option<Word>) -> Word {
        let mut existing = self.definitions().inlined_at.iter();
        let existing = existing.find(|it| it.matches(func, line, parent));
        if let Some(existing) = existing {
            existing.id
        } else {
            let id = self.id();
            self.definitions().inlined_at.push(InlinedAt {
                id,
                func,
                line,
                parent,
            });
            id
        }
    }

    // Deduplicated debug_source
    fn debug_source(&mut self, file: String) -> SourceFile {
        let existing = self.definitions().source_files.get(&file);
        if let Some(existing) = existing {
            *existing
        } else {
            let file_id = self.debug_string(&file);

            self.source(SourceLanguage::Unknown, 1, Some(file_id), Some("dummy"));
            let source = self.void_debug(None, Instructions::DebugSource, [file_id]);
            let comp_unit = self.debug_compilation_unit(source);
            let source_file = SourceFile {
                id: source,
                compilation_unit: comp_unit,
            };
            self.definitions().source_files.insert(file, source_file);
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

    fn debug_function(&mut self, name: impl Into<String>) -> Word {
        let name: String = name.into();
        if let Some(func) = self.definitions().functions.get(&name).copied() {
            func.id
        } else {
            let name_id = self.debug_string(&name);
            let id = self.id();
            let entry_name = self.debug_info().name_str.clone();
            let source = self.definitions().functions[&entry_name].source;

            self.definitions().functions.insert(
                name,
                FunctionDefinition {
                    id,
                    name: name_id,
                    file_name: name_id,
                    source,
                    ..Default::default()
                },
            );
            id
        }
    }

    pub fn finish_debug(&mut self) {
        if let Some(debug_info) = self.debug_info.clone() {
            for function in debug_info.definitions.functions.values() {
                self.declare_debug_function(function);
            }

            for inlined_at in &debug_info.definitions.inlined_at {
                self.declare_inlined_at(inlined_at);
            }

            // Declare entry
            let entry_name = debug_info.name_str.clone();
            let entry_def = self.definitions().functions[&entry_name];
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

    pub fn debug_scope(&mut self) {
        if self.debug {
            let func = *self.current_function_def();
            let line = self.stack_top().line;
            let col = self.stack_top().col;

            self.line(func.file_name, line, col);
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

    #[track_caller]
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
