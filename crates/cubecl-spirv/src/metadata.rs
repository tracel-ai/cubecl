use cubecl_core::ir as core;
use cubecl_core::ir::Metadata;
use rspirv::spirv::Word;

use crate::{
    item::{Elem, Item},
    variable::Variable,
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_meta(&mut self, meta: Metadata, out: Option<core::Variable>) {
        let out = out.unwrap();
        match meta {
            Metadata::Length { var } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                self.length(&var, Some(&out));
            }
            Metadata::Stride { dim, var } => {
                let int_ty = Item::Scalar(Elem::Int(32, false));
                let int = self.type_int(32, 0);
                let position = match var.kind {
                    core::VariableKind::GlobalInputArray(id) => id as usize,
                    core::VariableKind::GlobalOutputArray(id) => {
                        self.state.inputs.len() + id as usize
                    }
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };

                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);
                let one = self.const_u32(1);

                let dim_id = self.read(&dim);
                let rank_2 = self.rank_2();

                let index = match position > 1 {
                    true => {
                        let position = self.const_u32(position as u32);
                        self.i_mul(int, None, position, rank_2).unwrap()
                    }
                    false => rank_2,
                };
                let index = match position > 0 {
                    true => self.i_add(int, None, index, dim_id).unwrap(),
                    false => dim_id,
                };
                let index = self.i_add(int, None, index, one).unwrap();
                let index = Variable::Raw(index, int_ty.clone());
                let info = Variable::Named {
                    id: self.state.named["info"],
                    item: int_ty,
                    is_array: true,
                };
                self.read_indexed_unchecked(&out, &info, &index);
            }
            Metadata::Shape { dim, var } => {
                let int_ty = Item::Scalar(Elem::Int(32, false));
                let int = self.type_int(32, 0);
                let position = match var.kind {
                    core::VariableKind::GlobalInputArray(id) => id as usize,
                    core::VariableKind::GlobalOutputArray(id) => {
                        self.state.inputs.len() + id as usize
                    }
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);
                let one = self.const_u32(1);

                let dim_id = self.read(&dim);
                let rank = self.rank();
                let rank_2 = self.rank_2();
                let index = match position > 1 {
                    true => {
                        let position = self.const_u32(position as u32);
                        self.i_mul(int, None, position, rank_2).unwrap()
                    }
                    false => rank_2,
                };
                let index = match position > 0 {
                    true => self.i_add(int, None, index, rank).unwrap(),
                    false => rank,
                };
                let index = self.i_add(int, None, index, dim_id).unwrap();
                let index = self.i_add(int, None, index, one).unwrap();
                let index = Variable::Raw(index, int_ty.clone());
                let info = Variable::Named {
                    id: self.state.named["info"],
                    item: int_ty,
                    is_array: true,
                };
                self.read_indexed_unchecked(&out, &info, &index);
            }
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
            Variable::Slice {
                const_len: Some(len),
                ..
            } => {
                let len = self.const_u32(*len);
                if out.is_some() {
                    self.copy_object(out_ty, out_id, len).unwrap()
                } else {
                    len
                }
            }
            Variable::Slice { offset, end, .. } => {
                let len_ty = Elem::Int(32, false).id(self);
                self.i_sub(len_ty, out_id, *end, *offset).unwrap()
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
