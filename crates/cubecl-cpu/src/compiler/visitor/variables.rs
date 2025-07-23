use std::{collections::HashMap, num::NonZero};

use cubecl_core::ir::{
    Builtin, ConstantScalarValue, FloatKind, IntKind, Item, UIntKind, VariableKind,
};
use tracel_llvm::melior::{
    dialect::{
        index, memref,
        ods::{arith, vector},
    },
    ir::{
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::{IntegerType, MemRefType},
    },
};

use super::prelude::*;

#[derive(Default, Debug)]
pub struct Variables<'a> {
    pub local: HashMap<u32, Value<'a, 'a>>,
    pub version: HashMap<(u32, u16), Value<'a, 'a>>,
    pub mutable: HashMap<u32, Value<'a, 'a>>,
    pub global_constant: HashMap<u32, Item>,
}

impl<'a> Variables<'a> {
    pub fn new(opt: &Optimizer) -> Self {
        let mut variables = Self::default();
        for const_array in opt.const_arrays() {
            variables
                .global_constant
                .insert(const_array.id, const_array.item);
        }
        variables
    }
}

impl<'a> Visitor<'a> {
    pub fn insert_variable(&mut self, variable: Variable, value: Value<'a, 'a>) {
        match variable.kind {
            VariableKind::LocalConst { id } => {
                self.variables.local.insert(id, value);
            }
            VariableKind::Versioned { id, version } => {
                self.variables.version.insert((id, version), value);
            }
            VariableKind::LocalMut { id } => {
                let r#type = variable.elem().to_type(self.context);
                let memref_type = MemRefType::new(
                    r#type,
                    &[variable.vectorization_factor() as i64],
                    None,
                    None,
                );
                let memref = self.variables.mutable.get(&id).copied().unwrap_or_else(|| {
                    let value = self.append_operation_with_result(memref::alloca(
                        self.context,
                        memref_type,
                        &[],
                        &[],
                        None,
                        self.location,
                    ));
                    self.variables.mutable.insert(id, value);
                    value
                });
                let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
                let zero = self.append_operation_with_result(arith::constant(
                    self.context,
                    Type::index(self.context),
                    integer,
                    self.location,
                ));
                let operation = if value.r#type().is_vector() {
                    vector::store(self.context, value, memref, &[zero], self.location).into()
                } else {
                    memref::store(value, memref, &[zero], self.location)
                };
                self.block.append_operation(operation);
            }
            _ => todo!("This variable is not implemented {:?}", variable),
        };
    }
    pub fn get_binary_op_variable(
        &self,
        lhs: Variable,
        rhs: Variable,
    ) -> (Value<'a, 'a>, Value<'a, 'a>) {
        let vectorization_factor =
            std::cmp::max(lhs.vectorization_factor(), rhs.vectorization_factor());
        let (mut lhs_value, mut rhs_value) = (self.get_variable(lhs), self.get_variable(rhs));

        if lhs_value.r#type().is_vector() || rhs_value.r#type().is_vector() {
            if !lhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    lhs.elem().to_type(self.context),
                );
                lhs_value = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    lhs_value,
                    self.location,
                ));
            }
            if !rhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    rhs.elem().to_type(self.context),
                );
                rhs_value = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    rhs_value,
                    self.location,
                ));
            }
        }
        (lhs_value, rhs_value)
    }
    pub fn get_memory(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.args_manager.buffers[id as usize]
            }
            VariableKind::LocalMut { id } => *self
                .variables
                .mutable
                .get(&id)
                .expect("Variable should have been declared before"),
            VariableKind::ConstantArray { id, length } => {
                let name = id.to_string();
                let r#type = self
                    .variables
                    .global_constant
                    .get(&id)
                    .unwrap()
                    .to_type(self.context);
                let r#type = MemRefType::new(r#type, &[length as i64], None, None);
                self.append_operation_with_result(memref::get_global(
                    self.context,
                    &name,
                    r#type,
                    self.location,
                ))
            }
            _ => todo!(
                "This variable isn't backed by memory or implemented: {}",
                variable
            ),
        }
    }
    pub fn is_memory(&self, variable: Variable) -> bool {
        matches!(
            variable.kind,
            VariableKind::GlobalInputArray(_)
                | VariableKind::GlobalOutputArray(_)
                | VariableKind::LocalMut { .. }
                | VariableKind::ConstantArray { .. }
        )
    }
    pub fn get_variable(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::LocalConst { id } => *self
                .variables
                .local
                .get(&id)
                .expect("Variable should have been declared before"),
            VariableKind::Builtin(builtin) => self.get_builtin(builtin),
            VariableKind::ConstantScalar(constant_scalar_value) => {
                let (const_type, attribute) = match constant_scalar_value {
                    ConstantScalarValue::Int(value, int_kind) => {
                        let size = match int_kind {
                            IntKind::I8 => 8,
                            IntKind::I16 => 16,
                            IntKind::I32 => 32,
                            IntKind::I64 => 64,
                        };
                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute = IntegerAttribute::new(integer_type, value).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantScalarValue::UInt(value, int_kind) => {
                        let size = match int_kind {
                            UIntKind::U8 => 8,
                            UIntKind::U16 => 16,
                            UIntKind::U32 => 32,
                            UIntKind::U64 => 64,
                        };
                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute =
                            IntegerAttribute::new(integer_type, value as i64).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantScalarValue::Float(value, float_kind) => {
                        let float_type = match float_kind {
                            FloatKind::F16 => Type::float16(self.context),
                            FloatKind::BF16 => Type::bfloat16(self.context),
                            FloatKind::F32 => Type::float32(self.context),
                            FloatKind::F64 => Type::float64(self.context),
                            _ => panic!("Type is not supported in LLVM"),
                        };
                        let float_attribute =
                            FloatAttribute::new(self.context, float_type, value).into();
                        (float_type, float_attribute)
                    }
                    _ => todo!("Operation is not implemented {:?}", constant_scalar_value),
                };
                let value = self.append_operation_with_result(arith::constant(
                    self.context,
                    const_type,
                    attribute,
                    self.location,
                ));
                match variable.item.is_vectorized() {
                    true => {
                        let vector =
                            Type::vector(&[variable.vectorization_factor() as u64], const_type);
                        self.append_operation_with_result(vector::splat(
                            self.context,
                            vector,
                            value,
                            self.location,
                        ))
                    }
                    false => value,
                }
            }
            VariableKind::Versioned { id, version } => *self
                .variables
                .version
                .get(&(id, version))
                .expect("Variable should have been declared before"),
            VariableKind::LocalMut { id } => {
                let memref = *self
                    .variables
                    .mutable
                    .get(&id)
                    .expect("Variable should have been declared before");
                let result_type = variable.item.to_type(self.context);
                let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
                let zero = self.append_operation_with_result(arith::constant(
                    self.context,
                    Type::index(self.context),
                    integer,
                    self.location,
                ));
                if variable.item.is_vectorized() {
                    self.append_operation_with_result(vector::load(
                        self.context,
                        result_type,
                        memref,
                        &[zero],
                        self.location,
                    ))
                } else {
                    self.append_operation_with_result(memref::load(memref, &[zero], self.location))
                }
            }
            VariableKind::GlobalScalar(id) => {
                let memref = *self
                    .args_manager
                    .scalars_memref
                    .get(&variable.elem())
                    .unwrap();
                let index = self
                    .block
                    .const_int_from_type(
                        self.context,
                        self.location,
                        id as i64,
                        Type::index(self.context),
                    )
                    .unwrap();
                self.append_operation_with_result(memref::load(memref, &[index], self.location))
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
    pub fn get_index(&self, variable: Variable, target_item: Item) -> Value<'a, 'a> {
        let index = self.get_variable(variable);
        let mut index = self.append_operation_with_result(index::casts(
            index,
            Type::index(self.context),
            self.location,
        ));
        if target_item.is_vectorized() {
            let vectorization = target_item.vectorization.map(NonZero::get).unwrap_or(1u8) as i64;
            let shift = vectorization.ilog2() as i64;
            let constant = self.append_operation_with_result(arith::constant(
                self.context,
                Type::index(self.context),
                IntegerAttribute::new(Type::index(self.context), shift).into(),
                self.location,
            ));
            index = self.append_operation_with_result(arith::shli(
                self.context,
                index,
                constant,
                self.location,
            ));
        }
        index
    }

    pub fn get_builtin(&self, builtin: Builtin) -> Value<'a, 'a> {
        self.args_manager.get(builtin)
    }
}
