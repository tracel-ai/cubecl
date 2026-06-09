use cubecl_core::ir::{
    BinaryOperands, IndexOperands, Memory, Operator, StorageType, VectorInsertOperands,
};
use tracel_llvm::mlir_rs::{
    dialect::{
        arith, index, memref,
        ods::{self, llvm, vector},
    },
    ir::{
        Operation,
        r#type::{IntegerType, MemRefType, StridedLayoutAttr},
    },
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_memory(&mut self, memory: &Memory, out: Option<Variable>) {
        match memory {
            Memory::Reference(variable) => {
                let memref = self.get_memory(*variable);
                self.insert_variable(out.unwrap(), memref);
            }
            Memory::Index(index) => {
                let value = self.visit_index(index);
                self.insert_variable(out.unwrap(), value);
            }
            Memory::Load(variable) => {
                let out = out.unwrap();
                let memref = self.get_variable(*variable);
                let zero = self.create_constant_index(0);
                let operation = if out.ty.is_vectorized() {
                    let vector_type = Type::vector(
                        &[out.vector_size() as u64],
                        out.storage_type().to_type(self.context),
                    );
                    vector::load(self.context, vector_type, memref, &[zero], self.location).into()
                } else {
                    memref::load(memref, &[zero], self.location)
                };
                let value = self.append_operation_with_result(operation);
                self.insert_variable(out, value);
            }
            Memory::Store(op) => {
                let memref = self.get_variable(op.ptr);
                let value = self.get_variable(op.value);
                let zero = self.create_constant_index(0);
                let operation = if op.value.ty.is_vectorized() {
                    vector::store(self.context, value, memref, &[zero], self.location).into()
                } else {
                    memref::store(value, memref, &[zero], self.location)
                };
                self.block.append_operation(operation);
            }
            Memory::CopyMemory(op) => {
                assert_eq!(op.len, 1, "Bulk copy not supported on CPU");
                let source = self.get_variable(op.source);
                let target = self.get_variable(op.target);
                self.block
                    .append_operation(memref::copy(source, target, self.location));
            }
        }
    }

    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
            Operator::And(and) => {
                let lhs = self.get_variable(and.lhs);
                let rhs = self.get_variable(and.rhs);
                let value = self.append_operation_with_result(arith::andi(lhs, rhs, self.location));
                self.insert_variable(out, value);
            }
            Operator::Cast(cast) => {
                self.visit_cast(cast.input, out);
            }
            Operator::InitVector(init_vector) => {
                let inputs: Vec<_> = init_vector
                    .inputs
                    .iter()
                    .map(|input| self.get_variable(*input))
                    .collect();
                let result = out.ty.to_type(self.context);
                let init_vector = self.append_operation_with_result(vector::from_elements(
                    self.context,
                    result,
                    &inputs,
                    self.location,
                ));
                self.insert_variable(out, init_vector);
            }
            Operator::ExtractComponent(op) => {
                let value = self.visit_extract(op, out);
                self.insert_variable(out, value);
            }
            Operator::InsertComponent(op) => self.visit_insert(op, out),
            Operator::Not(not) => {
                let lhs = self.get_variable(not.input);
                let mask = self.create_int_constant_from_item(not.input.ty, -1);
                let value =
                    self.append_operation_with_result(arith::xori(lhs, mask, self.location));
                self.insert_variable(out, value);
            }
            Operator::Or(or) => {
                let lhs = self.get_variable(or.lhs);
                let rhs = self.get_variable(or.rhs);
                let value = self.append_operation_with_result(arith::ori(lhs, rhs, self.location));
                self.insert_variable(out, value);
            }
            Operator::Reinterpret(reinterpret) => {
                let target_type = out.ty.to_type(self.context);
                let input = self.get_variable(reinterpret.input);
                let value = self.append_operation_with_result(arith::bitcast(
                    input,
                    target_type,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
            Operator::Select(select) => {
                let condition = self.get_variable(select.cond);
                let condition = self.cast_to_bool(condition, select.cond.ty);
                let mut then = self.get_variable(select.then);
                let mut or_else = self.get_variable(select.or_else);
                if out.ty.is_vectorized() && !select.then.ty.is_vectorized() {
                    let vector = Type::vector(
                        &[out.vector_size() as u64],
                        select.then.storage_type().to_type(self.context),
                    );
                    then = self.append_operation_with_result(vector::broadcast(
                        self.context,
                        vector,
                        then,
                        self.location,
                    ));
                }
                if out.ty.is_vectorized() && !select.or_else.ty.is_vectorized() {
                    let vector = Type::vector(
                        &[out.vector_size() as u64],
                        select.or_else.storage_type().to_type(self.context),
                    );
                    or_else = self.append_operation_with_result(vector::broadcast(
                        self.context,
                        vector,
                        or_else,
                        self.location,
                    ));
                }
                let value = self.append_operation_with_result(arith::select(
                    condition,
                    then,
                    or_else,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
        }
    }

    fn visit_extract(&mut self, op: &BinaryOperands, out: Variable) -> Value<'a, 'a> {
        let mut index = self.get_variable(op.rhs);
        let u32_int = IntegerType::new(self.context, 32).into();
        if index.r#type() != u32_int {
            index = self.append_operation_with_result(index::casts(index, u32_int, self.location));
        }
        let to_extract = self.get_variable(op.lhs);
        let res = out.ty.to_type(self.context);
        let vector_extract =
            llvm::extractelement(self.context, res, to_extract, index, self.location);
        self.append_operation_with_result(vector_extract)
    }

    fn visit_index(&mut self, index: &IndexOperands) -> Value<'a, 'a> {
        assert!(index.vector_size == 0);
        let ty = index.list.ty;
        let index_value = self.get_index(index.index, ty, ty.is_vectorized());
        let memref = self.get_memory(index.list);

        let value_ty = index.list.storage_type().to_type(self.context);
        let layout = StridedLayoutAttr::new(self.context, i64::MIN, &[1]);
        let memref_ty = MemRefType::new(
            value_ty,
            &[ty.vector_size() as i64],
            Some(layout.into()),
            None,
        );

        let view = memref::subview(
            self.context,
            memref,
            &[index_value],
            &[],
            &[],
            &[i64::MIN],
            &[ty.vector_size() as i64],
            &[1],
            memref_ty,
            self.location,
        );
        self.append_operation_with_result(view)
    }

    fn visit_insert(&mut self, op: &VectorInsertOperands, out: Variable) {
        let mut index = self.get_variable(op.index);
        let index_ty = Type::index(self.context);
        if index.r#type() != index_ty {
            index = self.append_operation_with_result(index::casts(index, index_ty, self.location));
        }
        let memref = self.get_memory(out);
        let to_insert = self.get_variable(op.value);
        let vector_insert = memref::store(to_insert, memref, &[index], self.location);
        self.block.append_operation(vector_insert);
    }

    pub(crate) fn visit_cast(&mut self, to_cast: Variable, out: Variable) {
        let mut value = self.get_variable(to_cast);
        let target = out.ty.to_type(self.context);

        if !to_cast.ty.is_vectorized() && out.ty.is_vectorized() {
            let r#type = to_cast.storage_type().to_type(self.context);
            let vector_type = Type::vector(&[out.vector_size() as u64], r#type);
            value = self.append_operation_with_result(vector::broadcast(
                self.context,
                vector_type,
                value,
                self.location,
            ));
        };

        let value = if to_cast.storage_type().is_int() == out.storage_type().is_int() {
            self.get_cast_same_type_category(
                to_cast.storage_type(),
                out.storage_type(),
                target,
                value,
            )
        } else {
            self.get_cast_different_type_category(
                to_cast.storage_type(),
                out.storage_type(),
                target,
                value,
            )
        };
        self.insert_variable(out, value);
    }

    pub(crate) fn get_cast_different_type_category(
        &self,
        to_cast: StorageType,
        out: StorageType,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Value<'a, 'a> {
        if to_cast.is_int() {
            self.append_operation_with_result(self.cast_int_to_float(to_cast, target, value))
        } else {
            self.append_operation_with_result(self.cast_float_to_int(out, target, value))
        }
    }

    fn cast_float_to_int(
        &self,
        out: StorageType,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Operation<'a> {
        if out.is_signed_int() {
            arith::fptosi(value, target, self.location)
        } else {
            arith::fptoui(value, target, self.location)
        }
    }

    fn cast_int_to_float(
        &self,
        to_cast: StorageType,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Operation<'a> {
        if to_cast.is_signed_int() {
            arith::sitofp(value, target, self.location)
        } else {
            arith::uitofp(value, target, self.location)
        }
    }

    fn get_cast_same_type_category(
        &self,
        to_cast: StorageType,
        out: StorageType,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Value<'a, 'a> {
        if to_cast.size() > out.size() {
            self.append_operation_with_result(self.get_trunc(to_cast, target, value))
        } else if to_cast.size() < out.size() {
            self.append_operation_with_result(self.get_ext(to_cast, target, value))
        } else {
            value
        }
    }

    fn get_trunc(
        &self,
        to_cast: StorageType,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Operation<'a> {
        if to_cast.is_int() {
            arith::trunci(value, target, self.location)
        } else {
            ods::arith::truncf(self.context, target, value, self.location).into()
        }
    }

    fn get_ext(
        &self,
        to_cast: StorageType,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Operation<'a> {
        if to_cast.is_signed_int() {
            arith::extsi(value, target, self.location)
        } else if to_cast.is_unsigned_int() {
            arith::extui(value, target, self.location)
        } else {
            arith::extf(value, target, self.location)
        }
    }
}
