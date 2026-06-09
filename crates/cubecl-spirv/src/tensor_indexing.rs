use cubecl_core::ir::{self as core, ClampMode, TensorIndexingOps};
use rspirv::spirv::Capability;

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_tensor_indexing(&mut self, op: TensorIndexingOps, out: Option<core::Variable>) {
        self.capabilities.insert(Capability::TensorAddressingNV);
        let out = self.compile_variable(out.unwrap());
        match op {
            TensorIndexingOps::CreateLayout {
                shape,
                strides,
                clamp_mode,
            } => {
                let out_id = self.write_id(&out);
                let ty = out.item().id(self);

                let shape = shape
                    .into_iter()
                    .map(|it| self.compile_variable(it))
                    .collect::<Vec<_>>();
                let shape = shape.iter().map(|it| self.read(it)).collect::<Vec<_>>();

                let strides = strides.map(|s| {
                    s.into_iter()
                        .map(|it| self.compile_variable(it))
                        .collect::<Vec<_>>()
                });
                let strides = strides
                    .as_ref()
                    .map(|s| s.iter().map(|it| self.read(it)).collect::<Vec<_>>());

                let mut layout = self.create_tensor_layout_nv(ty, None).unwrap();
                // Write straight to out if strides and clamp is default
                let result_id = match (&strides, clamp_mode) {
                    (None, clamp_mode) if !matches!(clamp_mode, ClampMode::Constant(_)) => {
                        Some(out_id)
                    }
                    (None, ClampMode::Constant(0)) => Some(out_id),
                    _ => None,
                };
                layout = self
                    .tensor_layout_set_dimension_nv(ty, result_id, layout, shape)
                    .unwrap();
                let result_id = match clamp_mode {
                    ClampMode::Constant(0) => Some(out_id),
                    ClampMode::Constant(_) => None,
                    _ => Some(out_id),
                };
                if let Some(strides) = strides {
                    layout = self
                        .tensor_layout_set_stride_nv(ty, result_id, layout, strides)
                        .unwrap();
                }
                match clamp_mode {
                    ClampMode::Constant(0) => {}
                    ClampMode::Constant(val) => {
                        let val = self.const_u32(val);
                        layout = self
                            .tensor_layout_set_clamp_value_nv(ty, Some(out_id), layout, val)
                            .unwrap();
                    }
                    _ => {}
                }
                self.write(&out, layout);
            }
            TensorIndexingOps::CreateView => {
                let out_id = self.write_id(&out);
                let ty = out.item().id(self);

                self.create_tensor_view_nv(ty, Some(out_id)).unwrap();
                self.write(&out, out_id);
            }
            TensorIndexingOps::Slice {
                layout,
                offsets,
                shape,
            } => {
                let out_id = self.write_id(&out);
                let ty = out.item().id(self);

                let layout = self.compile_variable(layout);
                let layout = self.read(&layout);
                let args = offsets
                    .into_iter()
                    .zip(shape)
                    .flat_map(|(offset, shape)| {
                        let offset = self.compile_variable(offset);
                        let shape = self.compile_variable(shape);
                        [offset, shape]
                    })
                    .collect::<Vec<_>>();
                let args = args.iter().map(|it| self.read(it)).collect::<Vec<_>>();
                self.tensor_layout_slice_nv(ty, Some(out_id), layout, args)
                    .unwrap();
                self.write(&out, out_id);
            }
        }
    }
}
