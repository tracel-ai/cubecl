use cubecl_core::ir::{self as core, Scope};
use std::{
    collections::HashSet,
    fmt::Debug,
    mem::{take, transmute},
    ops::{Deref, DerefMut},
};

use cubecl_core::{
    ir::{HybridAllocator, KernelDefinition, LocalAllocator},
    Compiler, ExecutionMode,
};
use rspirv::{
    dr::{Builder, Module},
    spirv::{BuiltIn, Capability, Decoration, FunctionControl, StorageClass, Word},
};

use crate::{
    item::{Elem, Item},
    lookups::{ConstArray, LookupTables},
    target::{GLCompute, SpirvTarget},
    SpirvKernel,
};

pub struct SpirvCompiler<Target: SpirvTarget = GLCompute> {
    pub target: Target,
    builder: Builder,

    pub mode: ExecutionMode,
    global_invocation_id: Word,
    num_workgroups: Word,
    variable_block: usize,

    pub capabilities: HashSet<Capability>,
    pub state: LookupTables,
}

impl<T: SpirvTarget> Clone for SpirvCompiler<T> {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            builder: Builder::new_from_module(self.module_ref().clone()),
            mode: self.mode,
            global_invocation_id: self.global_invocation_id,
            num_workgroups: self.num_workgroups,
            variable_block: self.variable_block,

            capabilities: self.capabilities.clone(),
            state: self.state.clone(),
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
            variable_block: Default::default(),
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

    fn compile(kernel: KernelDefinition, mode: ExecutionMode) -> Self::Representation {
        let num_bindings = kernel.inputs.len() + kernel.outputs.len() + kernel.named.len();
        let module = Self {
            mode,
            ..Default::default()
        }
        .compile_kernel(kernel);
        SpirvKernel {
            module,
            num_bindings,
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
    pub fn compile_kernel(&mut self, kernel: KernelDefinition) -> Module {
        self.set_version(1, 6);

        self.init_state(kernel.clone());
        let cube_dims = vec![kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];

        let mut target = self.target.clone();
        let extensions = target.extensions(self);
        self.state.extensions = extensions;

        let void = self.type_void();
        let voidf = self.type_function(void, vec![]);
        let main = self
            .begin_function(void, None, FunctionControl::NONE, voidf)
            .unwrap();

        self.begin_block(None).unwrap();
        self.variable_block = self.selected_block().unwrap();
        self.select_block(None).unwrap(); // Pop variables so we can terminate it later once we're done

        let setup = self.id();
        let body = self.id();
        self.setup(setup, body);
        self.compile_scope(kernel.body, Some(body));
        self.ret().unwrap();

        // Terminate variable block
        let var_block = self.variable_block;
        self.select_block(Some(var_block)).unwrap();
        self.branch(setup).unwrap();

        self.end_function().unwrap();

        self.copy_const_arrays();

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

        take(&mut self.builder).module()
    }

    fn setup(&mut self, label: Word, body: Word) {
        self.begin_block(Some(label)).unwrap();
        let int = Item::Scalar(Elem::Int(32, false));
        let int_ty = int.id(self);
        let int_ptr = Item::Pointer(StorageClass::StorageBuffer, Box::new(int)).id(self);
        let info = self.state.named["info"];
        let zero = self.const_u32(0);
        let rank_ptr = self
            .access_chain(int_ptr, None, info, vec![zero, zero])
            .unwrap();
        self.state.rank = self.load(int_ty, None, rank_ptr, None, vec![]).unwrap();
        self.branch(body).unwrap();
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

    pub fn compile_scope(&mut self, mut scope: Scope, label: Option<Word>) -> Word {
        self.declare_const_arrays(scope.const_arrays.drain(..));

        let processed = scope.process();
        let label = self.begin_block(label).unwrap();

        for variable in processed.variables {
            let item = self.compile_item(variable.item());
            match variable {
                core::Variable::Local { id, depth, .. } => {
                    let ptr =
                        Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);

                    let var = self.declare_function_variable(ptr);
                    self.state.variables.insert((id, depth), var);
                }
                core::Variable::Slice { .. } => {}
                var => todo!("{var:?}"),
            };
        }

        for operation in processed.operations {
            self.compile_operation(operation);
        }
        label
    }

    // Declare variable in the first block of the function
    pub fn declare_function_variable(&mut self, ty: Word) -> Word {
        let current_block = self.selected_block();
        let var_block = self.variable_block;
        self.select_block(Some(var_block)).unwrap();
        let var = self.variable(ty, None, StorageClass::Function, None);
        self.select_block(current_block).unwrap();
        var
    }

    fn declare_const_arrays(
        &mut self,
        const_arrays: impl Iterator<Item = (core::Variable, Vec<core::Variable>)>,
    ) {
        let const_arrays = const_arrays
            .map(|(var, values)| {
                let item = self.compile_item(var.item());
                let len = values.len() as u32;
                let arr_ty = Item::Array(Box::new(item.clone()), len).id(self);
                let values: Vec<_> = values
                    .into_iter()
                    .map(|val| self.static_cast(val, &item))
                    .collect();
                let composite_id = self.constant_composite(arr_ty, values);
                ConstArray {
                    id: self.id(),
                    item,
                    len,
                    composite_id,
                }
            })
            .collect::<Vec<_>>();
        self.state.const_arrays.extend(const_arrays);
    }

    fn static_cast(&mut self, val: core::Variable, item: &Item) -> Word {
        let val = val.as_const().unwrap();
        let value = match (val, item.elem()) {
            (core::ConstantScalarValue::Int(val, _), Elem::Bool) => (val == 1) as u64,
            (core::ConstantScalarValue::Int(val, _), Elem::Int(64, _)) => unsafe {
                transmute::<i64, u64>(val)
            },
            (core::ConstantScalarValue::Int(val, _), Elem::Int(_, _)) => unsafe {
                transmute::<i64, u64>(val) as u32 as u64 // truncate
            },
            (core::ConstantScalarValue::Int(val, _), Elem::Float(64)) => (val as f64).to_bits(),
            (core::ConstantScalarValue::Int(val, _), Elem::Float(32)) => {
                (val as f32).to_bits() as u64
            }
            (core::ConstantScalarValue::Float(val, _), Elem::Bool) => (val == 1.0) as u64,
            (core::ConstantScalarValue::Float(val, _), Elem::Int(64, _)) => unsafe {
                transmute::<i64, u64>(val as i64)
            },
            (core::ConstantScalarValue::Float(val, _), Elem::Int(_, _)) => unsafe {
                transmute::<i64, u64>(val as i64) as u32 as u64 // truncate
            },
            (core::ConstantScalarValue::Float(val, _), Elem::Float(64)) => val.to_bits(),
            (core::ConstantScalarValue::Float(val, _), Elem::Float(_)) => {
                (val as f32).to_bits() as u64
            }
            (core::ConstantScalarValue::UInt(val), Elem::Bool) => (val == 1) as u64,
            (core::ConstantScalarValue::UInt(val), Elem::Int(64, false)) => val,
            (core::ConstantScalarValue::UInt(val), Elem::Int(64, true)) => val as i64 as u64, //clip sign bit
            (core::ConstantScalarValue::UInt(val), Elem::Int(_, false)) => val as u32 as u64, //truncate
            (core::ConstantScalarValue::UInt(val), Elem::Int(_, true)) => val as i32 as u64, //clip sign bit
            (core::ConstantScalarValue::UInt(val), Elem::Float(64)) => (val as f64).to_bits(),
            (core::ConstantScalarValue::UInt(val), Elem::Float(_)) => (val as f32).to_bits() as u64,
            (core::ConstantScalarValue::Bool(val), Elem::Bool) => val as u64,
            (core::ConstantScalarValue::Bool(val), Elem::Int(_, _)) => val as u64,
            (core::ConstantScalarValue::Bool(val), Elem::Float(64)) => {
                (val as u32 as f64).to_bits()
            }
            (core::ConstantScalarValue::Bool(val), Elem::Float(_)) => {
                (val as u32 as f32).to_bits() as u64
            }
            _ => unreachable!(),
        };
        println!(
            "val: {val:?}, value: {value}, value as float: {}, item: {item:?}",
            f32::from_bits(value as u32)
        );
        item.constant(self, value)
    }

    fn copy_const_arrays(&mut self) {
        let const_arrays = self.state.const_arrays.clone();

        for array in const_arrays {
            let ptr_ty = Item::Pointer(
                StorageClass::UniformConstant,
                Box::new(Item::Array(Box::new(array.item), array.len)),
            )
            .id(self);
            self.variable(
                ptr_ty,
                Some(array.id),
                StorageClass::UniformConstant,
                Some(array.composite_id),
            );
        }
    }
}
