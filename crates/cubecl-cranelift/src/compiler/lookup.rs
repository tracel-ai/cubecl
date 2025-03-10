use cranelift::prelude::{
    types, AbiParam, EntityRef, Type as CraneType, Variable as CraneliftVariable,
};
use cubecl_core::{
    compute::Binding,
    ir::{Elem, FloatKind, IntKind, Item as CubeItem, UIntKind, VariableKind},
};
use hashbrown::HashMap;

#[derive(Clone)]
pub(crate) struct LookupTables {
    func_counter: u32,
    var_counter: u32,
    ///map from function id to function name, used to create UserFuncName instances
    functions: HashMap<u32, String>,
    //TODO: need to change this to remove KernelVar enum, replace in/out with global{Input,Output}Array,
    //and change the key type to VariableKind
    ///map from indices in the kernel def to to cranelift Variables
    variables: HashMap<VariableKind, CraneliftVariable>,
}

///CubeIR variables have a id/index per variable kind, Cranelift expects a more global index
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum KernelVar {
    In(CubeItem),
    Out(CubeItem),
}

impl Default for LookupTables {
    fn default() -> Self {
        Self {
            func_counter: 0,
            var_counter: 0,
            functions: HashMap::new(),
            variables: HashMap::new(),
        }
    }
}
impl LookupTables {
    pub(crate) fn insert_func(&mut self, name: String) -> (u32, u32) {
        let id = self.func_counter;
        self.functions.insert(id, name);
        self.func_counter += 1;
        //the zero is the namespace id
        (0, id)
    }
    pub(crate) fn get_func(&self, id: u32) -> Option<&String> {
        self.functions.get(&id)
    }

    pub(crate) fn get(&self, var: VariableKind) -> Option<&CraneliftVariable> {
        todo!()
        //self.variables.get(&var)
    }

    pub(crate) fn getsert_var(&mut self, var: VariableKind) -> CraneliftVariable {
        *self.variables.entry(var).or_insert({
            let id = self.var_counter;
            self.var_counter += 1;
            CraneliftVariable::new(id as usize)
        })
    }
}

pub fn to_ssa_type(item: &CubeItem) -> CraneType {
    let mut t = match item.elem {
        Elem::Float(FloatKind::F16) => types::F16,
        Elem::Float(FloatKind::F32 | FloatKind::TF32 | FloatKind::Flex32) => types::F32,
        Elem::Float(FloatKind::F64) => types::F64,
        //https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/ir.md#integer-types
        //integer types can be interpreted as signed or unsigned.
        Elem::Int(IntKind::I8) | Elem::UInt(UIntKind::U8) | Elem::Bool => types::I8,
        Elem::Int(IntKind::I16) | Elem::UInt(UIntKind::U16) => types::I16,
        Elem::Int(IntKind::I32) | Elem::UInt(UIntKind::U32) => types::I32,
        Elem::Int(IntKind::I64) | Elem::UInt(UIntKind::U64) => types::I64,

        t => panic!("Unimplemented function parameter type {:?}", t),
    };
    if let Some(size) = item.vectorization {
        t = t.by(size.get() as u32).unwrap();
    }
    t
}

pub fn compile_binding(binding: &Binding) -> AbiParam {
    let vtype = to_ssa_type(&binding.item);

    AbiParam {
        value_type: vtype,
        purpose: cranelift_codegen::ir::ArgumentPurpose::Normal,
        extension: cranelift_codegen::ir::ArgumentExtension::None,
    }
}
