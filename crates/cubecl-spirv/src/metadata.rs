use cubecl_core::ir as core;
use cubecl_core::ir::Metadata;
use rspirv::spirv::{StorageClass, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::Variable,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_meta(&mut self, meta: Metadata, out: Option<core::Variable>, uniform: bool) {
        let out = out.unwrap();
        match meta {
            Metadata::Rank { var } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                let pos = self.ext_pos(&var);

                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);

                let offset = self.metadata.rank_index(pos);
                self.load_const_metadata(offset, Some(out_id));
                self.write(&out, out_id);
            }
            Metadata::Length { var } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                self.length(&var, Some(&out), uniform);
            }
            Metadata::BufferLength { var } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                self.buffer_length(&var, Some(&out), uniform);
            }
            Metadata::Stride { dim, var } => {
                let int = self.type_int(32, 0);

                let var = self.compile_variable(var);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);

                let out_id = out.id(self);
                self.mark_uniformity(out_id, uniform);

                let pos = self.ext_pos(&var);

                let offs_offset = self.metadata.stride_offset_index(pos);
                let offset = self.load_const_metadata(offs_offset, None);
                let dim_id = self.read(&dim);

                let index = self.i_add(int, None, offset, dim_id).unwrap();
                self.mark_uniformity(index, uniform);
                let index = Variable::Id(index);
                self.load_dyn_metadata(&index, &out);
            }
            Metadata::Shape { dim, var } => {
                let int = self.type_int(32, 0);

                let var = self.compile_variable(var);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);

                let out_id = out.id(self);
                self.mark_uniformity(out_id, uniform);

                let pos = self.ext_pos(&var);

                let offs_offset = self.metadata.shape_offset_index(pos);
                let offset = self.load_const_metadata(offs_offset, None);
                let dim_id = self.read(&dim);

                let index = self.i_add(int, None, offset, dim_id).unwrap();
                let index = Variable::Id(index);
                self.load_dyn_metadata(&index, &out);
            }
        }
    }

    pub fn length(&mut self, var: &Variable, out: Option<&Variable>, uniform: bool) -> Word {
        let (out_id, out_ty) = if let Some(out) = out {
            let out_id = self.write_id(out);
            self.mark_uniformity(out_id, uniform);
            let out_ty = out.elem().id(self);
            (Some(out_id), out_ty)
        } else {
            (None, self.type_int(32, 0))
        };

        let id = match var {
            Variable::GlobalInputArray(_, _, pos) | Variable::GlobalOutputArray(_, _, pos) => {
                let offset = self.metadata.len_index(*pos);
                let id = self.load_const_metadata(offset, out_id);

                if let Some(out_id) = out_id {
                    self.debug_name(out_id, format!("len({pos})"));
                }
                id
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

    pub fn buffer_length(&mut self, var: &Variable, out: Option<&Variable>, uniform: bool) -> Word {
        let out_id = out.map(|it| self.write_id(it));
        if let Some(out_id) = out_id {
            self.mark_uniformity(out_id, uniform);
        }

        let position = match var {
            Variable::GlobalInputArray(_, _, pos) | Variable::GlobalOutputArray(_, _, pos) => *pos,
            _ => panic!("Only Input and Output have a buffer length, got: {var:?}"),
        };
        let offset = self.metadata.buffer_len_index(position);
        let id = self.load_const_metadata(offset, out_id);

        if let Some(out) = out {
            self.debug_name(out_id.unwrap(), format!("buffer_len({position})"));
            self.write(out, id);
        }
        id
    }

    pub fn load_const_metadata(&mut self, index: u32, out: Option<Word>) -> Word {
        self.insert_global(|b| {
            let int = Item::Scalar(Elem::Int(32, false));
            let int_ty = int.id(b);
            let int_ptr = Item::Pointer(StorageClass::StorageBuffer, Box::new(int)).id(b);
            let info = b.state.info;
            let zero = b.const_u32(0);
            let index = b.const_u32(index);
            let info_ptr = b
                .access_chain(int_ptr, None, info, vec![zero, index])
                .unwrap();
            b.load(int_ty, out, info_ptr, None, vec![]).unwrap()
        })
    }

    pub fn load_dyn_metadata(&mut self, index: &Variable, out: &Variable) -> Word {
        let int_ty = Item::Scalar(Elem::Int(32, false));
        let info = Variable::Named {
            id: self.state.info,
            item: int_ty,
            is_array: false,
        };
        self.read_indexed_unchecked(out, &info, index)
    }

    fn ext_pos(&self, var: &Variable) -> u32 {
        let pos = match var {
            Variable::GlobalInputArray(_, _, pos) | Variable::GlobalOutputArray(_, _, pos) => *pos,
            _ => panic!("Only global buffers have rank"),
        };
        self.ext_meta_pos[pos as usize]
    }
}
