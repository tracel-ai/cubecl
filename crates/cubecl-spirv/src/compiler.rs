use cubecl_common::ExecutionMode;
use cubecl_core::{Metadata, WgpuCompilationOptions, ir as core, prelude::FastMath};
use cubecl_opt::{BasicBlock, NodeIndex, Optimizer, OptimizerBuilder, Uniformity};
use cubecl_runtime::config::{GlobalConfig, compilation::CompilationLogLevel};
use std::{
    collections::HashSet,
    fmt::Debug,
    mem::take,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use cubecl_core::{Compiler, compute::KernelDefinition};
use rspirv::{
    dr::{self, Builder, InsertPoint, Instruction, Module, Operand},
    spirv::{self, BuiltIn, Capability, Decoration, FPFastMathMode, Op, StorageClass, Word},
};

use crate::{
    SpirvKernel,
    debug::DebugInfo,
    item::Item,
    lookups::LookupTables,
    target::{GLCompute, SpirvTarget},
    transformers::{BitwiseTransform, ErfTransform},
};

pub struct SpirvCompiler<Target: SpirvTarget = GLCompute> {
    pub target: Target,
    pub(crate) builder: Builder,

    pub mode: ExecutionMode,
    pub debug_symbols: bool,
    pub fp_math_mode: FPFastMathMode,
    global_invocation_id: Word,
    num_workgroups: Word,
    pub setup_block: usize,
    pub opt: Rc<Optimizer>,
    pub uniformity: Rc<Uniformity>,
    pub current_block: Option<NodeIndex>,
    pub visited: HashSet<NodeIndex>,

    pub capabilities: HashSet<Capability>,
    pub float_controls: bool,
    pub state: LookupTables,
    pub ext_meta_pos: Vec<u32>,
    pub metadata: Metadata,
    pub debug_info: Option<DebugInfo>,
    compilation_options: WgpuCompilationOptions,
}

unsafe impl<T: SpirvTarget> Send for SpirvCompiler<T> {}
unsafe impl<T: SpirvTarget> Sync for SpirvCompiler<T> {}

impl<T: SpirvTarget> Clone for SpirvCompiler<T> {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            builder: Builder::new_from_module(self.module_ref().clone()),
            mode: self.mode,
            global_invocation_id: self.global_invocation_id,
            num_workgroups: self.num_workgroups,
            setup_block: self.setup_block,
            opt: self.opt.clone(),
            uniformity: self.uniformity.clone(),
            current_block: self.current_block,

            capabilities: self.capabilities.clone(),
            float_controls: false,
            state: self.state.clone(),
            debug_symbols: self.debug_symbols,
            fp_math_mode: self.fp_math_mode,
            visited: self.visited.clone(),
            metadata: self.metadata.clone(),
            debug_info: self.debug_info.clone(),
            ext_meta_pos: self.ext_meta_pos.clone(),
            compilation_options: self.compilation_options.clone(),
        }
    }
}

fn debug_symbols_activated() -> bool {
    matches!(
        GlobalConfig::get().compilation.logger.level,
        CompilationLogLevel::Full
    )
}

impl<T: SpirvTarget> Default for SpirvCompiler<T> {
    fn default() -> Self {
        Self {
            target: Default::default(),
            builder: Builder::new(),
            mode: Default::default(),
            global_invocation_id: Default::default(),
            num_workgroups: Default::default(),
            capabilities: Default::default(),
            float_controls: Default::default(),
            state: Default::default(),
            setup_block: Default::default(),
            opt: Default::default(),
            uniformity: Default::default(),
            current_block: Default::default(),
            debug_symbols: debug_symbols_activated(),
            fp_math_mode: FPFastMathMode::NONE,
            visited: Default::default(),
            metadata: Default::default(),
            debug_info: Default::default(),
            ext_meta_pos: Default::default(),
            compilation_options: Default::default(),
        }
    }
}

impl<T: SpirvTarget> Deref for SpirvCompiler<T> {
    type Target = Builder;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<T: SpirvTarget> DerefMut for SpirvCompiler<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

impl<T: SpirvTarget> Compiler for SpirvCompiler<T> {
    type Representation = SpirvKernel;
    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        value: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        let bindings = value.buffers.clone();
        let scalars = value
            .scalars
            .iter()
            .map(|s| (self.compile_elem(s.elem), s.count))
            .collect();
        let mut ext_meta_pos = Vec::new();
        let mut num_ext = 0;

        let mut all_meta: Vec<_> = value
            .buffers
            .iter()
            .map(|buf| (buf.id, buf.has_extended_meta))
            .chain(value.tensor_maps.iter().map(|id| (*id, true)))
            .collect();
        all_meta.sort_by_key(|(id, _)| *id);

        let num_meta = all_meta.len();

        for (_, has_extended_meta) in all_meta.iter() {
            ext_meta_pos.push(num_ext);
            if *has_extended_meta {
                num_ext += 1;
            }
        }

        self.mode = mode;
        self.metadata = Metadata::new(num_meta as u32, num_ext);
        self.compilation_options = compilation_options.clone();
        self.ext_meta_pos = ext_meta_pos;

        let (module, optimizer) = self.compile_kernel(value);
        SpirvKernel {
            module,
            optimizer,
            bindings,
            scalars,
            has_metadata: self.metadata.static_len() > 0,
        }
    }

    fn elem_size(&self, elem: core::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "spv"
    }
}

impl<Target: SpirvTarget> Debug for SpirvCompiler<Target> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spirv<{:?}>", self.target)
    }
}

impl<Target: SpirvTarget> SpirvCompiler<Target> {
    pub fn compile_kernel(&mut self, kernel: KernelDefinition) -> (Module, Optimizer) {
        let options = kernel.options.clone();

        self.debug_symbols = debug_symbols_activated() || options.debug_symbols;
        self.fp_math_mode = match self.compilation_options.supports_fp_fast_math {
            true => convert_math_mode(options.fp_math_mode),
            false => FPFastMathMode::NONE,
        };

        if self.fp_math_mode != FPFastMathMode::NONE {
            let inst = dr::Instruction::new(
                spirv::Op::Capability,
                None,
                None,
                vec![dr::Operand::LiteralBit32(6029)],
            );
            self.module_mut().capabilities.push(inst);
        }

        self.set_version(1, 6);

        let mut target = self.target.clone();

        self.init_state(kernel.clone());

        let mut opt = OptimizerBuilder::default()
            .with_transformer(ErfTransform)
            .with_transformer(BitwiseTransform)
            .optimize(kernel.body, kernel.cube_dim, self.mode);

        self.uniformity = opt.analysis::<Uniformity>();
        self.opt = Rc::new(opt);

        self.init_debug();

        let cube_dims = vec![kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];

        target.set_kernel_name(options.kernel_name.clone());

        let (main, debug_setup) = self.declare_main(&options.kernel_name);

        let setup = self.id();
        self.debug_name(setup, "setup");

        let entry = self.opt.entry();
        let body = self.label(entry);
        let setup_block = self.setup(setup, debug_setup);
        self.setup_block = setup_block;
        self.compile_block(entry);

        let ret = self.opt.ret;
        self.compile_block(ret);

        if self.selected_block().is_some() {
            let label = self.label(ret);
            self.branch(label).unwrap();
        }

        self.select_block(Some(setup_block)).unwrap();
        self.branch(body).unwrap();

        self.end_function().unwrap();

        self.declare_shared_memories();

        let builtins = self
            .state
            .used_builtins
            .clone()
            .into_iter()
            .map(|(builtin, (id, item))| {
                let ty = Item::Pointer(StorageClass::Input, Box::new(item)).id(self);
                self.variable(ty, Some(id), StorageClass::Input, None);
                self.decorate(id, Decoration::BuiltIn, vec![builtin.into()]);
                id
            })
            .collect::<Vec<_>>();

        target.set_modes(self, main, builtins, cube_dims);

        let module = take(&mut self.builder).module();
        (module, self.opt.as_ref().clone())
    }

    fn setup(&mut self, label: Word, debug_setup: impl Fn(&mut Self)) -> usize {
        self.begin_block(Some(label)).unwrap();

        let opt = self.opt.clone();
        for const_arr in opt.const_arrays() {
            self.register_const_array(const_arr);
        }

        debug_setup(self);

        let setup_block = self.selected_block().unwrap();
        self.select_block(None).unwrap();
        setup_block
    }

    #[track_caller]
    pub fn current_block(&self) -> BasicBlock {
        self.opt.block(self.current_block.unwrap()).clone()
    }

    pub fn builtin(&mut self, builtin: BuiltIn, item: Item) -> Word {
        if let Some(existing) = self.state.used_builtins.get(&builtin) {
            existing.0
        } else {
            let id = self.id();
            self.state.used_builtins.insert(builtin, (id, item));
            id
        }
    }

    pub fn compile_block(&mut self, block: NodeIndex) {
        if self.visited.contains(&block) {
            return;
        }
        self.visited.insert(block);
        self.current_block = Some(block);

        let label = self.label(block);
        self.begin_block(Some(label)).unwrap();
        let block_id = self.selected_block().unwrap();

        self.debug_start_block();

        let operations = self.current_block().ops.borrow().clone();
        for (_, operation) in operations {
            self.compile_operation(operation);
        }

        let control_flow = self.current_block().control_flow.borrow().clone();
        self.compile_control_flow(control_flow);

        let current = self.selected_block();
        self.select_block(Some(block_id)).unwrap();
        let phi = { self.opt.block(block).phi_nodes.borrow().clone() };
        for phi in phi {
            let out = self.compile_variable(phi.out);
            let ty = out.item().id(self);
            let out_id = self.write_id(&out);
            let entries: Vec<_> = phi
                .entries
                .into_iter()
                .map(|it| {
                    let label = self.end_label(it.block);
                    let value = self.compile_variable(it.value);
                    let value = self.read(&value);
                    (value, label)
                })
                .collect();
            self.insert_phi(InsertPoint::Begin, ty, Some(out_id), entries)
                .unwrap();
        }
        self.select_block(current).unwrap();
    }

    // Declare variable in the first block of the function
    pub fn declare_function_variable(&mut self, ty: Word) -> Word {
        let setup = self.setup_block;
        let id = self.id();
        let var = Instruction::new(
            Op::Variable,
            Some(ty),
            Some(id),
            vec![Operand::StorageClass(StorageClass::Function)],
        );
        let current_block = self.selected_block();
        self.select_block(Some(setup)).unwrap();
        self.insert_into_block(InsertPoint::Begin, var).unwrap();
        self.select_block(current_block).unwrap();
        id
    }

    fn declare_shared_memories(&mut self) {
        let shared_memories = self.state.shared_memories.clone();
        for (_, memory) in shared_memories {
            let arr_ty = Item::Array(Box::new(memory.item), memory.len);
            let ptr_ty = Item::Pointer(StorageClass::Workgroup, Box::new(arr_ty)).id(self);

            self.debug_var_name(memory.id, memory.var);
            self.variable(ptr_ty, Some(memory.id), StorageClass::Workgroup, None);
        }
    }

    pub fn declare_float_execution_modes(&mut self, main: Word) {
        let mode = self.const_u32(self.fp_math_mode.bits());

        let types = self.builder.module_ref().types_global_values.clone();
        let scalars = types
            .iter()
            .filter(|inst| inst.class.opcode == Op::TypeFloat)
            .map(|it| it.result_id.expect("OpTypeFloat always has result ID"))
            .collect::<Vec<_>>();
        for ty in scalars {
            let operands = vec![
                dr::Operand::IdRef(main),
                dr::Operand::LiteralBit32(6028),
                dr::Operand::LiteralBit32(ty),
                dr::Operand::LiteralBit32(mode),
            ];

            let inst = dr::Instruction::new(spirv::Op::ExecutionModeId, None, None, operands);
            self.module_mut().execution_modes.push(inst);
        }
    }

    pub fn is_uniform_block(&self) -> bool {
        self.uniformity
            .is_block_uniform(self.current_block.unwrap())
    }
}

fn convert_math_mode(math_mode: FastMath) -> FPFastMathMode {
    let mut flags = FPFastMathMode::NONE;

    for mode in math_mode.iter() {
        match mode {
            FastMath::NotNaN => flags |= FPFastMathMode::NOT_NAN,
            FastMath::NotInf => flags |= FPFastMathMode::NOT_INF,
            FastMath::UnsignedZero => flags |= FPFastMathMode::NSZ,
            FastMath::AllowReciprocal => flags |= FPFastMathMode::ALLOW_RECIP,
            FastMath::AllowContraction => flags |= FPFastMathMode::from_bits_retain(0x10000),
            FastMath::AllowReassociation => flags |= FPFastMathMode::from_bits_retain(0x20000),
            FastMath::AllowTransform => {
                flags |= FPFastMathMode::from_bits_retain(0x10000)
                    | FPFastMathMode::from_bits_retain(0x20000)
                    | FPFastMathMode::from_bits_retain(0x40000)
            }
            _ => {}
        }
    }

    flags
}
