use cubecl_core::ir as core;
use cubecl_core::ir::Metadata;
use rspirv::spirv::Word;

use crate::{
    item::{Elem, Item},
    variable::Variable,
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_meta(&mut self, meta: Metadata) {
        match meta {
            Metadata::Length { var, out } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                self.length(&var, Some(&out));
            }
            Metadata::Stride { dim, var, out } => {
                let int_ty = Item::Scalar(Elem::Int(32, false));
                let int = self.type_int(32, 0);
                let position = match var {
                    core::Variable::GlobalInputArray { id, .. } => id as usize,
                    core::Variable::GlobalOutputArray { id, .. } => {
                        self.state.inputs.len() + id as usize
                    }
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                let position = self.const_u32(position as u32);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);
                let one = self.const_u32(1);

                let dim_id = self.read(&dim);
                let out_id = self.write_id(&out);
                let rank_2 = self.state.rank_2;
                let index = self.i_mul(int, None, position, rank_2).unwrap();
                let index = self.i_add(int, None, index, dim_id).unwrap();
                let index = self.i_add(int, None, index, one).unwrap();
                let index = Variable::LocalBinding {
                    id: index,
                    item: int_ty.clone(),
                };
                let info = Variable::Named {
                    id: self.state.named["info"],
                    item: int_ty,
                    is_array: true,
                };
                self.read_indexed(out_id, &info, &index);
            }
            Metadata::Shape { dim, var, out } => {
                let int_ty = Item::Scalar(Elem::Int(32, false));
                let int = self.type_int(32, 0);
                let position = match var {
                    core::Variable::GlobalInputArray { id, .. } => id as usize,
                    core::Variable::GlobalOutputArray { id, .. } => {
                        self.state.inputs.len() + id as usize
                    }
                    _ => panic!("Only Input and Output have a stride, got: {:?}", var),
                };
                let position = self.const_u32(position as u32);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);
                let one = self.const_u32(1);

                let dim_id = self.read(&dim);
                let out_id = self.write_id(&out);
                let rank = self.state.rank;
                let rank_2 = self.state.rank_2;
                let index = self.i_mul(int, None, position, rank_2).unwrap();
                let index = self.i_add(int, None, index, rank).unwrap();
                let index = self.i_add(int, None, index, dim_id).unwrap();
                let index = self.i_add(int, None, index, one).unwrap();
                let index = Variable::LocalBinding {
                    id: index,
                    item: int_ty.clone(),
                };
                let info = Variable::Named {
                    id: self.state.named["info"],
                    item: int_ty,
                    is_array: true,
                };
                self.read_indexed(out_id, &info, &index);
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
