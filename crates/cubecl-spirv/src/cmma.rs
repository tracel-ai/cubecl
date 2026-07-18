// use crate::{
//     SpirvCompiler, SpirvTarget,
//     item::{Elem, Item},
//     lookups::Matrix,
//     value::Value,
// };
// use cubecl_core::ir::{self as core, ElemType};
// use rspirv::{
//     dr::Operand,
//     spirv::{
//         self, Capability, CooperativeMatrixLayout, CooperativeMatrixOperands, CooperativeMatrixUse,
//         MemoryAccess, TensorAddressingOperands,
//     },
// };

// impl<T: SpirvTarget> SpirvCompiler<T> {
//     pub fn compile_cmma(&mut self, cmma: CoopMma, out: Option<core::ExpandValue>) {

//         match cmma {
//             CoopMma::LoadTensor {
//                 buffer,
//                 layout,
//                 view,
//             } => self.compile_load_tensor(out.unwrap(), buffer, layout, view),
//             CoopMma::StoreTensor { mat, layout, view } => {
//                 self.compile_store_tensor(mat, out.unwrap(), layout, view)
//             }

//         }
//     }

//     fn compile_load_tensor(
//         &mut self,
//         mat: core::ExpandValue,
//         buffer: core::ExpandValue,
//         layout: core::ExpandValue,
//         view: Option<core::ExpandValue>,
//     ) {
//         self.capabilities
//             .insert(Capability::CooperativeMatrixTensorAddressingNV);

//         let mat = self.compile_value(mat);
//         let write_id = self.write_id_cmma(&mat);

//         let buffer = self.compile_value(buffer);
//         let layout = self.compile_value(layout);
//         let view = view.map(|view| self.compile_value(view));
//         let layout = self.read(&layout);
//         let view = view.map(|view| self.read(&view));

//         let ptr = buffer.id(self);
//         let out_ty = mat.item().unwrap_ptr();
//         let align = buffer.item().value_type().size();
//         let ty = out_ty.id(self);

//         let zero = Item::Scalar(mat.elem()).const_u32(self, 0);
//         let clipped_fallback = self.composite_construct(ty, None, [zero]).unwrap();

//         let (operands, extra_args) = match view {
//             Some(view) => (
//                 TensorAddressingOperands::TENSOR_VIEW,
//                 vec![Operand::IdRef(view)],
//             ),
//             None => (TensorAddressingOperands::NONE, vec![]),
//         };

//         let mat_id = self
//             .cooperative_matrix_load_tensor_nv(
//                 ty,
//                 Some(write_id),
//                 ptr,
//                 clipped_fallback,
//                 layout,
//                 MemoryAccess::ALIGNED,
//                 [align.into()],
//                 operands,
//                 extra_args,
//             )
//             .unwrap();

//         self.write_cmma(&mat, mat_id);
//     }

//     fn compile_store_tensor(
//         &mut self,
//         mat: core::ExpandValue,
//         out: core::ExpandValue,
//         layout: core::ExpandValue,
//         view: Option<core::ExpandValue>,
//     ) {
//         self.capabilities
//             .insert(Capability::CooperativeMatrixTensorAddressingNV);

//         let mat = self.compile_value(mat);
//         let mat_obj = self.read(&mat);
//         //assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

//         let out = self.compile_value(out);
//         let layout = self.compile_value(layout);
//         let view = view.map(|view| self.compile_value(view));

//         let layout = self.read(&layout);
//         let view = view.map(|view| self.read(&view));

//         let align = out.item().value_type().size();
//         let ptr = out.id(self);

//         let (operands, extra_args) = match view {
//             Some(view) => (
//                 TensorAddressingOperands::TENSOR_VIEW,
//                 vec![Operand::IdRef(view)],
//             ),
//             None => (TensorAddressingOperands::NONE, vec![]),
//         };

//         self.cooperative_matrix_store_tensor_nv(
//             ptr,
//             mat_obj,
//             layout,
//             MemoryAccess::ALIGNED,
//             [align.into()],
//             operands,
//             extra_args,
//         )
//         .unwrap();
//     }
