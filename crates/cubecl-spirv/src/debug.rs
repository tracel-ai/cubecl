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

#[derive(Clone, Debug)]
pub struct DebugInfo {
    pub source: Word,
    pub name: Word,
    pub compilation_unit: Word,
    pub entry_point: Word,
    pub functions: Vec<(Word, Option<Word>)>,
    strings: HashMap<String, Word>,
    function_defs: HashMap<String, Word>,
    inlined_at_defs: HashMap<(Word, Option<Word>), Word>,
    function_ty: Word,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_debug(&mut self, kernel: KernelDefinition) {
        if self.debug {
            let mut strings = HashMap::new();
            let name = self.string(&kernel.kernel_name);
            strings.insert(kernel.kernel_name, name);

            self.source(SourceLanguage::Unknown, 1, Some(name), Some("source"));
            let source = self.void_debug(Instructions::DebugSource, [name]);
            let version = self.const_u32(1);
            let dwarf = self.const_u32(4);
            let lang = self.const_u32(SourceLanguage::Unknown as u32);
            let compilation_unit = self.void_debug(
                Instructions::DebugCompilationUnit,
                [version, dwarf, source, lang],
            );
            let source = self.void_debug(Instructions::DebugSource, [name]);
            let flags = self.const_u32(DebugInfoFlags::None.0);
            let return_ty = self.type_void();
            let function_ty = self.void_debug(Instructions::DebugTypeFunction, [flags, return_ty]);

            self.debug_info = Some(DebugInfo {
                source,
                name,
                compilation_unit,
                entry_point: 0,
                functions: vec![],
                strings,
                inlined_at_defs: Default::default(),
                function_defs: Default::default(),
                function_ty,
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
            self.debug_info().functions.push((debug_function, None));
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
                core::DebugInfo::BeginCall { name } => {
                    let func = self.debug_function(name);
                    let (parent, inline_parent) = *self.debug_info().functions.last().unwrap();

                    let inlined_at = self.inlined_at(parent, inline_parent);

                    if self.is_last_instruction_scope() {
                        let _ = self.pop_instruction();
                    }

                    self.debug_info().functions.push((func, Some(inlined_at)));
                    self.void_op(Instructions::DebugScope, [func, inlined_at]);
                }
                core::DebugInfo::EndCall => {
                    self.debug_info().functions.pop();
                    if self.is_last_instruction_scope() {
                        let _ = self.pop_instruction();
                    }
                    let (func, _) = *self.debug_info().functions.last().unwrap();
                    self.void_op(Instructions::DebugScope, [func]);
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
    fn inlined_at(&mut self, parent: Word, inline_parent: Option<Word>) -> Word {
        let existing = self
            .debug_info()
            .inlined_at_defs
            .get(&(parent, inline_parent));
        if let Some(existing) = existing {
            *existing
        } else {
            let zero = self.const_u32(0);
            let inlined_at = if let Some(inline_parent) = inline_parent {
                self.void_debug(Instructions::DebugInlinedAt, [zero, parent, inline_parent])
            } else {
                self.void_debug(Instructions::DebugInlinedAt, [zero, parent])
            };
            self.debug_info()
                .inlined_at_defs
                .insert((parent, inline_parent), inlined_at);
            inlined_at
        }
    }

    fn debug_function(&mut self, name: impl Into<String>) -> Word {
        let name: String = name.into();
        if let Some(func) = self.debug_info().function_defs.get(&name).copied() {
            func
        } else {
            let name_id = self.debug_string(&name);
            let flags = self.const_u32(DebugInfoFlags::None.0);
            let debug_type = self.debug_info().function_ty;
            let source = self.debug_info().source;
            let compilation_unit = self.debug_info().compilation_unit;
            let zero = self.const_u32(0);
            let debug_func = self.void_debug(
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
            self.debug_info().function_defs.insert(name, debug_func);
            debug_func
        }
    }

    pub fn debug_scope(&mut self) {
        if self.debug {
            let (func, _) = *self.debug_info().functions.last().unwrap();
            let name = self.debug_info().name;
            self.line(name, 0, 0);
            self.void_op(Instructions::DebugScope, [func]);
        }
    }

    fn void_debug<const N: usize>(
        &mut self,
        instruction: Instructions,
        operands: [Word; N],
    ) -> Word {
        let ext = self.state.extensions[DEBUG_EXT_NAME];
        let void = self.type_void();
        let operands = operands.into_iter().map(Operand::IdRef).collect::<Vec<_>>();
        let out = self.id();
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
