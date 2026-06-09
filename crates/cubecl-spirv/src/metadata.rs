use cubecl_core::ir::Metadata;
use cubecl_core::ir::{self as core};
use rspirv::spirv::{MemoryAccess, Word};

use crate::{SpirvCompiler, SpirvTarget, item::Item, variable::Variable};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_meta(&mut self, meta: Metadata, out: Option<core::Variable>, uniform: bool) {
        let out = out.unwrap();
        match meta {
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

                let offs_offset = self.info.metadata.stride_offset_index(pos);
                let offset = self.load_const_metadata(offs_offset, None, out.item());
                let dim_id = self.read_as(&dim, &out.item());

                let index = self.i_add(ty_id, None, offset, dim_id).unwrap();
                self.mark_uniformity(index, uniform);
                let index = Variable::Raw(index, out.item());
                self.load_dyn_metadata(&index, Some(out_id), out.item());
                self.write(&out, out_id);
            }
            Metadata::Shape { dim, var } => {
                let var = self.compile_variable(var);
                let dim = self.compile_variable(dim);
                let out = self.compile_variable(out);

                let ty_id = out.item().id(self);
                let out_id = out.id(self);
                self.mark_uniformity(out_id, uniform);

                let pos = self.ext_pos(&var);

                let offs_offset = self.info.metadata.shape_offset_index(pos);
                let offset = self.load_const_metadata(offs_offset, None, out.item());
                let dim_id = self.read_as(&dim, &out.item());

                let index = self.i_add(ty_id, None, offset, dim_id).unwrap();
                let index = Variable::Id(index);
                self.load_dyn_metadata(&index, Some(out_id), out.item());
                self.write(&out, out_id);
            }
        }
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
            Variable::GlobalBuffer(_, _, pos) => *pos,
            _ => panic!("Only Input and Output have a buffer length, got: {var:?}"),
        };
        let offset = self.info.metadata.buffer_len_index(position);
        let id = self.load_const_metadata(offset, out_id, out_ty);

        if let Some(out) = out {
            self.debug_name(out_id.unwrap(), format!("buffer_len({position})"));
            self.write(out, id);
        }
        id
    }

    pub fn load_const_metadata(&mut self, index: u32, out: Option<Word>, ty: Item) -> Word {
        self.insert_in_setup(|b| {
            let align = ty.size();
            let ty_id = ty.id(b);
            let storage_class = T::info_storage_class(b);
            let ptr_ty = Item::Pointer(storage_class, Box::new(ty)).id(b);
            let info = b.state.info.unwrap().id;
            let offset = b.const_u32(b.state.scalar_bindings.len() as u32);
            let index = b.const_u32(index);
            let info_ptr = b
                .in_bounds_access_chain(ptr_ty, None, info, vec![offset, index])
                .unwrap();
            b.load(
                ty_id,
                out,
                info_ptr,
                Some(MemoryAccess::ALIGNED),
                [align.into()],
            )
            .unwrap()
        })
    }

    pub fn load_dyn_metadata(&mut self, index: &Variable, out: Option<Word>, ty: Item) -> Word {
        let align = ty.size();
        let ty_id = ty.id(self);
        let storage_class = T::info_storage_class(self);
        let ptr_ty = Item::Pointer(storage_class, Box::new(ty)).id(self);
        let info = self.state.info.unwrap().id;
        let offset = self.const_u32(self.state.scalar_bindings.len() as u32 + 1);
        let index = self.read(index);
        let info_ptr = self
            .in_bounds_access_chain(ptr_ty, None, info, vec![offset, index])
            .unwrap();
        self.load(
            ty_id,
            out,
            info_ptr,
            Some(MemoryAccess::ALIGNED),
            [align.into()],
        )
        .unwrap()
    }

    fn ext_pos(&self, var: &Variable) -> u32 {
        let pos = match var {
            Variable::GlobalBuffer(_, _, pos) => *pos,
            _ => panic!("Only global buffers have rank"),
        };
        self.ext_meta_pos[pos as usize]
    }
}
