use cubecl_core::ir as core;
use cubecl_core::ir::Metadata;
use rspirv::spirv::{StorageClass, Word};

use crate::{SpirvCompiler, SpirvTarget, item::Item, variable::Variable};

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
                self.load_const_metadata(offset, Some(out_id), out.item());
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
                let var = self.compile_variable(var);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);

                let ty_id = out.item().id(self);
                let out_id = out.id(self);
                self.mark_uniformity(out_id, uniform);

                let pos = self.ext_pos(&var);

                let offs_offset = self.metadata.stride_offset_index(pos);
                let offset = self.load_const_metadata(offs_offset, None, out.item());
                let dim_id = self.read_as(&dim, &out.item());

                let index = self.i_add(ty_id, None, offset, dim_id).unwrap();
                self.mark_uniformity(index, uniform);
                let index = Variable::Raw(index, out.item());
                self.load_dyn_metadata(&index, &out, out.item());
            }
            Metadata::Shape { dim, var } => {
                let var = self.compile_variable(var);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);

                let ty_id = out.item().id(self);
                let out_id = out.id(self);
                self.mark_uniformity(out_id, uniform);

                let pos = self.ext_pos(&var);

                let offs_offset = self.metadata.shape_offset_index(pos);
                let offset = self.load_const_metadata(offs_offset, None, out.item());
                let dim_id = self.read_as(&dim, &out.item());

                let index = self.i_add(ty_id, None, offset, dim_id).unwrap();
                let index = Variable::Id(index);
                self.load_dyn_metadata(&index, &out, out.item());
            }
        }
    }

    pub fn length(&mut self, var: &Variable, out: Option<&Variable>, uniform: bool) -> Word {
        let (out_id, out_ty) = if let Some(out) = out {
            let out_id = self.write_id(out);
            self.mark_uniformity(out_id, uniform);
            (Some(out_id), out.item())
        } else {
            (None, self.compile_type(self.addr_type.into()))
        };
        let ty_id = out_ty.id(self);

        let id = match var {
            Variable::GlobalInputArray(_, _, pos) | Variable::GlobalOutputArray(_, _, pos) => {
                let offset = self.metadata.len_index(*pos);
                let id = self.load_const_metadata(offset, out_id, out_ty);

                if let Some(out_id) = out_id {
                    self.debug_name(out_id, format!("len({pos})"));
                }
                id
            }
            Variable::Slice {
                const_len: Some(len),
                ..
            } => {
                let len = out_ty.const_u32(self, *len);
                if out.is_some() {
                    self.copy_object(ty_id, out_id, len).unwrap()
                } else {
                    len
                }
            }
            Variable::Slice { offset, end, .. } => {
                self.i_sub(ty_id, out_id, *end, *offset).unwrap()
            }
            Variable::SharedArray(_, _, len)
            | Variable::ConstantArray(_, _, len)
            | Variable::LocalArray(_, _, len) => out_ty.const_u32(self, *len),
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
        let out_ty = out
            .map(|it| it.item())
            .unwrap_or_else(|| self.compile_type(self.addr_type.into()));

        let position = match var {
            Variable::GlobalInputArray(_, _, pos) | Variable::GlobalOutputArray(_, _, pos) => *pos,
            _ => panic!("Only Input and Output have a buffer length, got: {var:?}"),
        };
        let offset = self.metadata.buffer_len_index(position);
        let id = self.load_const_metadata(offset, out_id, out_ty);

        if let Some(out) = out {
            self.debug_name(out_id.unwrap(), format!("buffer_len({position})"));
            self.write(out, id);
        }
        id
    }

    pub fn load_const_metadata(&mut self, index: u32, out: Option<Word>, ty: Item) -> Word {
        self.insert_in_setup(|b| {
            let ty_id = ty.id(b);
            let int_ptr = Item::Pointer(StorageClass::StorageBuffer, Box::new(ty)).id(b);
            let info = b.state.info;
            let zero = b.const_u32(0);
            let index = b.const_u32(index);
            let info_ptr = b
                .access_chain(int_ptr, None, info, vec![zero, index])
                .unwrap();
            b.load(ty_id, out, info_ptr, None, vec![]).unwrap()
        })
    }

    pub fn load_dyn_metadata(&mut self, index: &Variable, out: &Variable, ty: Item) -> Word {
        let info = Variable::Named {
            id: self.state.info,
            item: ty,
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
