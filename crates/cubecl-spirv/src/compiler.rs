use crate::{
    SpirvKernel,
    debug::DebugInfo,
    item::Item,
    lookups::CompilerState,
    target::{GLCompute, SpirvTarget},
    transformers::{BitwiseTransform, ErfTransform, HypotTransform, RhypotTransform},
};
use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    Compiler, CubeDim, Info, Metadata, WgpuCompilationOptions,
    ir::{self as core, ElemType, Id, InstructionModes, StorageType, UIntKind, features::EnumSet},
    post_processing::{
        checked_io::CheckedIoVisitor, disaggregate::DisaggregateVisitor,
        saturating::SaturatingArithmeticProcessor, unroll::UnrollVisitor,
    },
    prelude::{FastMath, KernelDefinition, Visibility},
    server::ExecutionMode,
};
use cubecl_opt::{
    BasicBlock, Function, NodeIndex, Optimizer, OptimizerBuilder, SharedLiveness, Uniformity,
};
use cubecl_runtime::{
    compiler::CompilationError,
    config::{CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel},
};
use rspirv::{
    binary::Assemble,
    dr::{Builder, InsertPoint, Instruction, Module, Operand},
    spirv::{BuiltIn, Capability, Decoration, FPFastMathMode, Op, StorageClass, Word},
};
use std::{
    collections::HashSet,
    fmt::Debug,
    mem::take,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Arc,
};

pub struct SpirvCompiler<Target: SpirvTarget = GLCompute> {
    pub target: Target,
    pub(crate) builder: Builder,

    pub cube_dim: CubeDim,
    pub mode: ExecutionMode,
    pub addr_type: StorageType,
    pub debug_symbols: bool,
    global_invocation_id: Word,
    num_workgroups: Word,
    pub setup_block: usize,
    pub opt: Rc<Optimizer>,
    pub uniformity: Rc<Uniformity>,
    pub shared_liveness: Rc<SharedLiveness>,
    pub current_func: Option<Id>,
    pub current_block: Option<NodeIndex>,

    pub capabilities: HashSet<Capability>,
    pub state: CompilerState,
    pub ext_meta_pos: Vec<u32>,
    pub info: Info,
    pub debug_info: Option<DebugInfo>,
    pub compilation_options: WgpuCompilationOptions,
}

unsafe impl<T: SpirvTarget> Send for SpirvCompiler<T> {}
unsafe impl<T: SpirvTarget> Sync for SpirvCompiler<T> {}

impl<T: SpirvTarget> Clone for SpirvCompiler<T> {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            builder: Builder::new_from_module(self.module_ref().clone()),
            cube_dim: self.cube_dim,
            mode: self.mode,
            addr_type: self.addr_type,
            global_invocation_id: self.global_invocation_id,
            num_workgroups: self.num_workgroups,
            setup_block: self.setup_block,
            opt: self.opt.clone(),
            uniformity: self.uniformity.clone(),
            shared_liveness: self.shared_liveness.clone(),
            current_func: self.current_func,
            current_block: self.current_block,
            capabilities: self.capabilities.clone(),
            state: self.state.clone(),
            debug_symbols: self.debug_symbols,
            info: self.info.clone(),
            debug_info: self.debug_info.clone(),
            ext_meta_pos: self.ext_meta_pos.clone(),
            compilation_options: self.compilation_options,
        }
    }
}

fn debug_symbols_activated() -> bool {
    matches!(
        CubeClRuntimeConfig::get().compilation.logger.level,
        CompilationLogLevel::Full
    )
}

impl<T: SpirvTarget> Default for SpirvCompiler<T> {
    fn default() -> Self {
        Self {
            target: Default::default(),
            builder: Builder::new(),
            cube_dim: CubeDim::new_single(),
            mode: Default::default(),
            addr_type: ElemType::UInt(UIntKind::U32).into(),
            global_invocation_id: Default::default(),
            num_workgroups: Default::default(),
            capabilities: Default::default(),
            state: Default::default(),
            setup_block: Default::default(),
            opt: Default::default(),
            uniformity: Default::default(),
            shared_liveness: Default::default(),
            current_func: Default::default(),
            current_block: Default::default(),
            debug_symbols: debug_symbols_activated(),
            info: Default::default(),
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
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = value.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile spirv kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        let bindings = value.buffers.clone();
        let mut ext_meta_pos = Vec::new();
        let mut num_ext = 0;

        let mut all_meta: Vec<_> = value
            .buffers
            .iter()
            .chain(value.tensor_maps.iter())
            .map(|buf| (buf.id, buf.has_extended_meta))
            .collect();
        all_meta.sort_by_key(|(id, _)| *id);

        let num_meta = all_meta.len();

        for (_, has_extended_meta) in all_meta.iter() {
            ext_meta_pos.push(num_ext);
            if *has_extended_meta {
                num_ext += 1;
            }
        }

        let metadata = Metadata::new(num_meta as u32, num_ext);

        self.cube_dim = value.cube_dim;
        self.mode = mode;
        self.addr_type = addr_type;
        self.info = Info::new(&value.scalars, metadata, addr_type);
        self.compilation_options = *compilation_options;
        self.ext_meta_pos = ext_meta_pos;

        let (module, optimizer, shared_size) = self.compile_kernel(value);
        let info_visibility = match T::info_storage_class(self) {
            StorageClass::Uniform => Visibility::Uniform,
            _ => Visibility::Read,
        };
        let immediate_size = match T::params_storage_class(self, bindings.len()) {
            StorageClass::PushConstant => Some((bindings.len() + 1) * size_of::<u64>()),
            _ => None,
        };

        let visibility = self.opt.global_state.buffer_visibility.borrow();
        let bindings = visibility
            .iter()
            .map(|vis| match vis.writable {
                true => Visibility::ReadWrite,
                false => Visibility::Read,
            })
            .collect();

        Ok(SpirvKernel {
            assembled_module: module.assemble(),
            module: Some(Arc::new(module)),
            optimizer: Some(Arc::new(optimizer)),
            bindings,
            shared_size,
            immediate_size,
            info_visibility,
        })
    }

    fn elem_size(&self, elem: core::ElemType) -> usize {
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
    pub fn compile_kernel(&mut self, mut kernel: KernelDefinition) -> (Module, Optimizer, usize) {
        let options = kernel.options.clone();

        self.debug_symbols = debug_symbols_activated() || options.debug_symbols;

        let version = self.compilation_options.vulkan.max_spirv_version;
        self.set_version(version.0, version.1);

        let mut target = self.target.clone();

        let mut opt = OptimizerBuilder::default()
            .with_transformer(ErfTransform)
            .with_transformer(BitwiseTransform::new(
                self.compilation_options.vulkan.supports_arbitrary_bitwise,
            ))
            .with_transformer(HypotTransform)
            .with_transformer(RhypotTransform)
            .with_visitor(CheckedIoVisitor::new(
                self.mode,
                kernel.options.kernel_name.clone(),
            ))
            .with_visitor(DisaggregateVisitor::default())
            .with_visitor(UnrollVisitor::new(
                self.compilation_options.vulkan.max_vector_size,
            ))
            .with_processor(SaturatingArithmeticProcessor::new(true))
            .optimize(kernel.body.clone(), kernel.cube_dim);

        self.uniformity = opt.main.analysis::<Uniformity>(&opt.global_state);
        self.shared_liveness = opt.main.analysis::<SharedLiveness>(&opt.global_state);
        self.opt = Rc::new(opt);

        self.init_debug();
        self.init_base_state(&mut kernel);
        let shared_size = self.declare_shared_memories();

        let cube_dims = vec![kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];

        target.set_kernel_name(options.kernel_name.clone());

        let opt = self.opt.clone();
        for (id, func) in opt.global_state.extra_functions.iter() {
            let def = self.declare_function(func);
            self.current_func = Some(*id);

            let blocks = func.breadth_first_dominators();
            for block in blocks {
                self.compile_block(block);
            }
            self.end_function_and_reset_lookups();

            self.state.extra_funcs.insert(*id, def);
            self.current_func = None;
        }

        self.init_kernel_state(kernel);

        let (main, debug_setup) = self.declare_main(&options.kernel_name);

        let setup = self.id();
        self.debug_name(setup, "setup");

        let entry = self.opt.entry();
        let body = self.label(entry);

        let setup_block = self.setup(setup, debug_setup);
        self.setup_block = setup_block;

        let blocks = opt.main.breadth_first_dominators();
        for block in blocks {
            self.compile_block(block);
        }

        self.select_block(Some(setup_block)).unwrap();
        self.branch(body).unwrap();

        // Don't reset the state here, need to keep used builtins around
        self.end_function().unwrap();

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
        (module, self.opt.as_ref().clone(), shared_size)
    }

    fn setup(&mut self, label: Word, debug_setup: impl Fn(&mut Self)) -> usize {
        self.begin_block(Some(label)).unwrap();

        let opt = self.opt.clone();
        for const_arr in opt.main.const_arrays() {
            self.register_const_array(const_arr);
        }

        Target::load_params(self);

        debug_setup(self);

        let setup_block = self.selected_block().unwrap();
        self.select_block(None).unwrap();
        setup_block
    }

    #[track_caller]
    pub fn current_block(&self) -> BasicBlock {
        self.current_func()
            .block(self.current_block.unwrap())
            .clone()
    }

    pub fn current_func(&self) -> &Function {
        self.current_func
            .map(|func| &self.opt.global_state.extra_functions[&func])
            .unwrap_or(&self.opt.main)
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
        let phi = { self.current_func().block(block).phi_nodes.borrow().clone() };
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
    pub fn declare_function_variable(&mut self, ty: Word, init: Option<Word>) -> Word {
        let setup = self.setup_block;
        let id = self.id();
        let mut var = Instruction::new(
            Op::Variable,
            Some(ty),
            Some(id),
            vec![Operand::StorageClass(StorageClass::Function)],
        );
        if let Some(init) = init {
            var.operands.push(Operand::IdRef(init));
        }
        let current_block = self.selected_block();
        self.select_block(Some(setup)).unwrap();
        self.insert_into_block(InsertPoint::Begin, var).unwrap();
        self.select_block(current_block).unwrap();
        id
    }

    fn declare_shared_memories(&mut self) -> usize {
        if self.compilation_options.vulkan.supports_explicit_smem {
            self.declare_shared_memories_explicit() as usize
        } else {
            self.declare_shared_memories_implicit() as usize
        }
    }

    /// When using `VK_KHR_workgroup_memory_explicit_layout`, all shared memory is declared as a
    /// `Block`. This means they are all pointers into the same chunk of memory, with different
    /// offsets and sizes. Unlike C++, this shared block is declared implicitly, not explicitly.
    /// Alignment and total size is calculated by the driver.
    fn declare_shared_memories_explicit(&mut self) -> u32 {
        let mut shared_size = 0;

        let shared = self.state.shared.clone();
        if shared.is_empty() {
            return shared_size;
        }

        self.capabilities
            .insert(Capability::WorkgroupMemoryExplicitLayoutKHR);

        for (index, memory) in shared {
            let memory_size = memory.item.size();
            let value_size = memory.item.value_type().size();
            shared_size = shared_size.max(memory.offset + memory_size);

            // It's safe to assume that if 8-bit/16-bit types are supported, they're supported for
            // explicit layout as well.
            match value_size {
                1 => {
                    self.capabilities
                        .insert(Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR);
                }
                2 => {
                    self.capabilities
                        .insert(Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
                }
                _ => {}
            }

            let item_id = memory.item.id(self);
            let block_id = self.id();

            if let Item::Array(_, _) = memory.item {
                self.decorate(item_id, Decoration::ArrayStride, [value_size.into()]);
            }

            self.type_struct_id(Some(block_id), [item_id]);

            self.decorate(block_id, Decoration::Block, []);
            self.member_decorate(block_id, 0, Decoration::Offset, [memory.offset.into()]);

            let ptr_ty =
                self.type_pointer(Some(memory.ptr_ty_id), StorageClass::Workgroup, block_id);

            self.debug_shared(memory.id, index);
            self.variable(ptr_ty, Some(memory.id), StorageClass::Workgroup, None);
            self.decorate(memory.id, Decoration::Aliased, []);
        }

        shared_size
    }

    fn declare_shared_memories_implicit(&mut self) -> u32 {
        let mut shared_size = 0;

        let shared = self.state.shared.clone();
        for (index, memory) in shared {
            shared_size += memory.item.size();

            let ty_id = memory.item.id(self);
            let ptr_ty = self.type_pointer(Some(memory.ptr_ty_id), StorageClass::Workgroup, ty_id);

            self.debug_shared(memory.id, index);
            self.variable(ptr_ty, Some(memory.id), StorageClass::Workgroup, None);
        }
        shared_size
    }

    pub fn declare_math_mode(&mut self, modes: InstructionModes, out_id: Word) {
        if !self.compilation_options.vulkan.supports_fp_fast_math || modes.fp_math_mode.is_empty() {
            return;
        }
        let mode = convert_math_mode(modes.fp_math_mode);
        self.capabilities.insert(Capability::FloatControls2);
        self.decorate(
            out_id,
            Decoration::FPFastMathMode,
            [Operand::FPFastMathMode(mode)],
        );
    }

    pub fn is_uniform_block(&self) -> bool {
        self.uniformity
            .is_block_uniform(self.current_block.unwrap())
    }
}

pub(crate) fn convert_math_mode(math_mode: EnumSet<FastMath>) -> FPFastMathMode {
    let mut flags = FPFastMathMode::NONE;

    for mode in math_mode.iter() {
        match mode {
            FastMath::NotNaN => flags |= FPFastMathMode::NOT_NAN,
            FastMath::NotInf => flags |= FPFastMathMode::NOT_INF,
            FastMath::UnsignedZero => flags |= FPFastMathMode::NSZ,
            FastMath::AllowReciprocal => flags |= FPFastMathMode::ALLOW_RECIP,
            FastMath::AllowContraction => flags |= FPFastMathMode::ALLOW_CONTRACT,
            FastMath::AllowReassociation => flags |= FPFastMathMode::ALLOW_REASSOC,
            FastMath::AllowTransform => {
                flags |= FPFastMathMode::ALLOW_CONTRACT
                    | FPFastMathMode::ALLOW_REASSOC
                    | FPFastMathMode::ALLOW_TRANSFORM
            }
            _ => {}
        }
    }

    flags
}
