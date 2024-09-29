use cubecl_core::ir::{self as core, Metadata};
use cubecl_core::ir::{Operation, Operator};
use rspirv::spirv::Word;

use crate::{containers::Slice, item::Elem, variable::Variable, SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, op: Operation) {
        match op {
            Operation::Operator(operator) => self.compile_operator(operator),
            Operation::Branch(branch) => self.compile_branch(branch),
            Operation::Metadata(meta) => self.compile_meta(meta),
            op => todo!("{op:?}"),
        }
    }

    pub fn compile_operator(&mut self, op: Operator) {
        match op {
            Operator::Equal(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let lhs_id = self.read(&lhs);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);
                let ty = out.item().id(self);
                match lhs.elem() {
                    Elem::Bool => self.logical_equal(ty, Some(out_id), lhs_id, rhs_id),
                    Elem::Int(_) => self.i_equal(ty, Some(out_id), lhs_id, rhs_id),
                    Elem::Float(_) => self.f_ord_equal(ty, Some(out_id), lhs_id, rhs_id),
                    Elem::Void => unreachable!(),
                }
                .unwrap();
                self.write(&out, out_id);
            }
            Operator::Index(op) => {
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_id = self.write_id(&out);

                self.read_indexed(out_id, &value, &index);
                self.write(&out, out_id);
            }
            Operator::IndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let value_id = self.read(&value);

                self.write_indexed(&out, &index, value_id);
            }
            Operator::Slice(op) => {
                let item = self.compile_item(op.input.item());
                let input = self.compile_variable(op.input);
                let start = self.compile_variable(op.start);
                let end = self.compile_variable(op.end);
                let out = match op.out {
                    core::Variable::Slice { id, depth, .. } => (id, depth),
                    _ => unreachable!(),
                };

                let start_id = self.read(&start);
                let end_id = self.read(&end);
                let len_ty = Elem::Int(32).id(self);
                let len = self.i_sub(len_ty, None, end_id, start_id).unwrap();
                self.state.slices.insert(
                    out,
                    Slice {
                        ptr: input,
                        offset: start_id,
                        len,
                        item,
                    },
                );
            }
            op => todo!("{op:?}"),
        }
    }

    fn compile_meta(&mut self, meta: Metadata) {
        match meta {
            Metadata::Length { var, out } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                self.length(&var, Some(&out));
            }
            meta => todo!("{meta:?}"),
        }
    }

    pub fn length(&mut self, var: &Variable, out: Option<&Variable>) -> Word {
        let (out_id, out_ty) = if let Some(out) = out {
            let out_id = self.write_id(out);
            let out_ty = out.elem().id(self);
            (Some(out_id), out_ty)
        } else {
            (None, self.type_int(32, 0))
        };

        let id = match var {
            Variable::GlobalInputArray(ptr, _) | Variable::GlobalOutputArray(ptr, _) => {
                self.array_length(out_ty, out_id, *ptr, 0).unwrap()
            }
            Variable::Slice { len, .. } => {
                if out.is_some() {
                    self.copy_object(out_ty, out_id, *len).unwrap()
                } else {
                    *len
                }
            }
            Variable::SharedMemory(_, _, len)
            | Variable::ConstantArray(_, _, len)
            | Variable::LocalArray(_, _, len) => self.const_u32(*len),
            var => unimplemented!("Var {var:?} doesn't have length"),
        };
        if let Some(out) = out {
            self.write(out, id);
        }
        id
    }
}
