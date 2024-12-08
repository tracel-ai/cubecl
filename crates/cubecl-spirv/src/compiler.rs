use cubecl_core::{ir as core, Metadata};
use cubecl_opt::{BasicBlock, NodeIndex, Optimizer};
use std::{
    collections::HashSet,
    env,
    fmt::Debug,
    mem::take,
    ops::{Deref, DerefMut},
};

use cubecl_core::{
    ir::{HybridAllocator, KernelDefinition, LocalAllocator},
    Compiler, ExecutionMode,
};
use rspirv::{
    dr::{Builder, InsertPoint, Instruction, Module, Operand},
    spirv::{BuiltIn, Capability, Decoration, Op, StorageClass, Word},
};

use crate::{
    debug::DebugInfo,
    item::Item,
    lookups::LookupTables,
    target::{GLCompute, SpirvTarget},
    SpirvKernel,
};

#[derive(Clone, Debug, Default)]
pub struct CompilationOptions {}

pub struct SpirvCompiler<Target: SpirvTarget = GLCompute> {
    pub target: Target,
    builder: Builder,

    pub mode: ExecutionMode,
    pub debug: bool,
    global_invocation_id: Word,
    num_workgroups: Word,
    pub setup_block: usize,
    pub opt: Optimizer,
    pub current_block: Option<NodeIndex>,
    pub visited: HashSet<NodeIndex>,

    pub capabilities: HashSet<Capability>,
    pub state: LookupTables,
    pub ext_meta_pos: Vec<u32>,
    pub metadata: Metadata,
    pub debug_info: Option<DebugInfo>,
    compilation_options: CompilationOptions,
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
            current_block: self.current_block,

            capabilities: self.capabilities.clone(),
            state: self.state.clone(),
            debug: self.debug,
            visited: self.visited.clone(),
            metadata: self.metadata.clone(),
            debug_info: self.debug_info.clone(),
            ext_meta_pos: self.ext_meta_pos.clone(),
            compilation_options: self.compilation_options.clone(),
        }
    }
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
            state: Default::default(),
            setup_block: Default::default(),
            opt: Default::default(),
            current_block: Default::default(),
            debug: env::var("CUBECL_DEBUG_LOG").is_ok(),
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
    type CompilationOptions = CompilationOptions;

    fn compile(
        value: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        let bindings = value
            .inputs
            .clone()
            .into_iter()
            .chain(value.outputs.clone())
            .chain(value.named.clone().into_iter().map(|it| it.1))
            .collect();
        let num_meta = value.inputs.len() + value.outputs.len();
        let mut ext_meta_pos = Vec::new();
        let mut num_ext = 0;

        for binding in value.inputs.iter().chain(value.outputs.iter()) {
            ext_meta_pos.push(num_ext);
            if binding.has_extended_meta {
                num_ext += 1;
            }
        }

        let (module, optimizer) = Self {
            mode,
            metadata: Metadata::new(num_meta as u32, num_ext),
            compilation_options: compilation_options.clone(),
            ext_meta_pos,
            ..Default::default()
        }
        .compile_kernel(value);
        SpirvKernel {
            module,
            optimizer,
            bindings,
        }
    }

    fn elem_size(elem: core::Elem) -> usize {
        elem.size()
    }

    fn local_allocator() -> impl LocalAllocator {
        HybridAllocator::default()
    }

    fn max_shared_memory_size() -> usize {
        32768
    }
}

impl<Target: SpirvTarget> Debug for SpirvCompiler<Target> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spirv<{:?}>", self.target)
    }
}

impl<Target: SpirvTarget> SpirvCompiler<Target> {
    pub fn compile_kernel(&mut self, kernel: KernelDefinition) -> (Module, Optimizer) {
        self.set_version(1, 6);

        let mut target = self.target.clone();
        let extensions = target.extensions(self);
        self.state.extensions = extensions;

        self.init_state(kernel.clone());
        self.init_debug(kernel.clone());
        let cube_dims = vec![kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];

        target.set_kernel_name(kernel.kernel_name.clone());

        let (main, debug_setup) = self.declare_main(&kernel.kernel_name);

        let setup = self.id();
        self.debug_name(setup, "setup");
        self.opt = Optimizer::new(kernel.body, kernel.cube_dim, self.mode);

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
        (module, self.opt.clone())
    }

    fn setup(&mut self, label: Word, debug_setup: impl Fn(&mut Self)) -> usize {
        self.begin_block(Some(label)).unwrap();

        for const_arr in self.opt.const_arrays() {
            self.register_const_array(const_arr);
        }

        debug_setup(self);

        let setup_block = self.selected_block().unwrap();
        self.select_block(None).unwrap();
        setup_block
    }

    #[track_caller]
    pub fn current_block(&self) -> &BasicBlock {
        self.opt.block(self.current_block.unwrap())
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

        self.debug_scope();

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
        for (id, memory) in shared_memories {
            let arr_ty = Item::Array(Box::new(memory.item), memory.len);
            let ptr_ty = Item::Pointer(StorageClass::Workgroup, Box::new(arr_ty)).id(self);

            self.debug_name(memory.id, format!("shared({id})"));
            self.variable(ptr_ty, Some(memory.id), StorageClass::Workgroup, None);
        }
    }

    pub fn debug_name(&mut self, var: Word, name: impl Into<String>) {
        if self.debug {
            self.name(var, name);
        }
    }
}
