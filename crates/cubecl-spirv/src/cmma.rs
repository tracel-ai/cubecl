use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    lookups::Matrix,
    variable::Value,
};
use cubecl_core::ir::{self as core, CoopMma, ElemType, Id, MatrixLayout, MatrixScope};
use rspirv::{
    dr::Operand,
    spirv::{
        self, Capability, CooperativeMatrixLayout, CooperativeMatrixOperands, CooperativeMatrixUse,
        MemoryAccess, TensorAddressingOperands,
    },
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_cmma(&mut self, cmma: CoopMma, out: Option<core::Value>) {
        self.capabilities.insert(Capability::CooperativeMatrixKHR);

        if let Some(out) = out {
            if out.elem_type() == ElemType::Float(core::FloatKind::BF16) {
                self.capabilities
                    .insert(Capability::BFloat16CooperativeMatrixKHR);
            }
            if matches!(
                out.elem_type(),
                ElemType::Float(core::FloatKind::E5M2 | core::FloatKind::E4M3)
            ) {
                self.capabilities
                    .insert(Capability::Float8CooperativeMatrixEXT);
            }
        }

        match cmma {
            CoopMma::Fill { value } => self.compile_fill(out.unwrap(), value),
            CoopMma::Load {
                ptr,
                stride,
                layout,
            } => self.compile_load(out.unwrap(), ptr, stride, layout),
            CoopMma::LoadTensor {
                buffer,
                layout,
                view,
            } => self.compile_load_tensor(out.unwrap(), buffer, layout, view),
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => self.compile_execute(mat_a, mat_b, mat_c, out.unwrap()),
            CoopMma::ExecuteElementwise { matrix, op } => {
                self.compile_elementwise_op(matrix, op, out.unwrap());
            }
            CoopMma::Store {
                mat,
                stride,
                layout,
                destination,
            } => self.compile_store(mat, stride, destination, layout),
            CoopMma::StoreTensor { mat, layout, view } => {
                self.compile_store_tensor(mat, out.unwrap(), layout, view)
            }
            CoopMma::Cast { input } => self.compile_cast(input, out.unwrap()),
            CoopMma::RowIndex { .. }
            | CoopMma::ColIndex { .. }
            | CoopMma::LoadMatrix { .. }
            | CoopMma::StoreMatrix { .. }
            | CoopMma::ExecuteManual { .. }
            | CoopMma::ExecuteScaled { .. } => {
                panic!("Manual register management not currently supported in SPIR-V")
            }
        }
    }

    fn compile_load(
        &mut self,
        mat: core::Value,
        ptr: core::Value,
        stride: core::Value,
        layout: Option<MatrixLayout>,
    ) {
        let mat = self.compile_value(mat);
        let write_id = self.write_id_cmma(&mat);

        let ptr = self.compile_value(ptr);
        let stride = self.compile_value(stride);
        let stride_item = stride.item();
        let mut stride = self.read(&stride);

        let value_ty = ptr.item().value_type();
        let align = value_ty.size();

        if let Item::Vector(_, vector_size) = value_ty {
            let shift = stride_item.const_u32(self, vector_size.trailing_zeros());
            let stride_ty = stride_item.id(self);
            stride = self
                .shift_right_logical(stride_ty, None, stride, shift)
                .unwrap();
        }

        let layout = layout
            .and_then(compile_layout)
            .or(matrix_layout(&mat))
            .unwrap_or(CooperativeMatrixLayout::RowMajorKHR);
        let memory_layout = self.const_u32(layout as u32);

        let ptr = self.read(&ptr);
        let out_ty = mat.item().unwrap_ptr();
        let ty = out_ty.id(self);

        let mat_id = self
            .cooperative_matrix_load_khr(
                ty,
                Some(write_id),
                ptr,
                memory_layout,
                Some(stride),
                Some(MemoryAccess::ALIGNED),
                [align.into()],
            )
            .unwrap();

        self.write_cmma(&mat, mat_id);
    }

    fn compile_load_tensor(
        &mut self,
        mat: core::Value,
        buffer: core::Value,
        layout: core::Value,
        view: Option<core::Value>,
    ) {
        self.capabilities
            .insert(Capability::CooperativeMatrixTensorAddressingNV);

        let mat = self.compile_value(mat);
        let write_id = self.write_id_cmma(&mat);

        let buffer = self.compile_value(buffer);
        let layout = self.compile_value(layout);
        let view = view.map(|view| self.compile_value(view));
        let layout = self.read(&layout);
        let view = view.map(|view| self.read(&view));

        let ptr = buffer.id(self);
        let out_ty = mat.item().unwrap_ptr();
        let align = buffer.item().value_type().size();
        let ty = out_ty.id(self);

        let zero = Item::Scalar(mat.elem()).const_u32(self, 0);
        let clipped_fallback = self.composite_construct(ty, None, [zero]).unwrap();

        let (operands, extra_args) = match view {
            Some(view) => (
                TensorAddressingOperands::TENSOR_VIEW,
                vec![Operand::IdRef(view)],
            ),
            None => (TensorAddressingOperands::NONE, vec![]),
        };

        let mat_id = self
            .cooperative_matrix_load_tensor_nv(
                ty,
                Some(write_id),
                ptr,
                clipped_fallback,
                layout,
                MemoryAccess::ALIGNED,
                [align.into()],
                operands,
                extra_args,
            )
            .unwrap();

        self.write_cmma(&mat, mat_id);
    }

    fn compile_fill(&mut self, mat: core::Value, value: core::Value) {
        let mat = self.compile_value(mat);
        let value = self.compile_value(value);
        let mat_id = self.write_id_cmma(&mat);

        let item = mat.item().unwrap_ptr();
        let ty = item.id(self);
        let mat_id = match value {
            Value::Constant(id, _, _) => self.constant_composite(ty, vec![id]),
            var => {
                let var = self.read(&var);
                self.composite_construct(ty, Some(mat_id), vec![var])
                    .unwrap()
            }
        };

        self.write_cmma(&mat, mat_id);
    }

    fn compile_store(
        &mut self,
        mat: core::Value,
        stride: core::Value,
        destination: core::Value,
        layout: MatrixLayout,
    ) {
        let mat = self.compile_value(mat);
        let mat_obj = self.read(&mat);
        //assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

        let ptr = self.compile_value(destination);
        let stride = self.compile_value(stride);
        let value_ty = ptr.item().value_type();

        let stride_item = stride.item();
        let mut stride = self.read(&stride);
        let layout = compile_layout(layout).unwrap_or(CooperativeMatrixLayout::RowMajorKHR);
        let memory_layout = self.const_u32(layout as u32);

        let ptr = self.read(&ptr);

        let align = value_ty.size();

        if let Item::Vector(_, vector_size) = value_ty {
            let shift = stride_item.const_u32(self, vector_size.trailing_zeros());
            let stride_ty = stride_item.id(self);
            stride = self
                .shift_right_logical(stride_ty, None, stride, shift)
                .unwrap();
        }

        self.cooperative_matrix_store_khr(
            ptr,
            mat_obj,
            memory_layout,
            Some(stride),
            Some(MemoryAccess::ALIGNED),
            [align.into()],
        )
        .unwrap();
    }

    fn compile_store_tensor(
        &mut self,
        mat: core::Value,
        out: core::Value,
        layout: core::Value,
        view: Option<core::Value>,
    ) {
        self.capabilities
            .insert(Capability::CooperativeMatrixTensorAddressingNV);

        let mat = self.compile_value(mat);
        let mat_obj = self.read(&mat);
        //assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

        let out = self.compile_value(out);
        let layout = self.compile_value(layout);
        let view = view.map(|view| self.compile_value(view));

        let layout = self.read(&layout);
        let view = view.map(|view| self.read(&view));

        let align = out.item().value_type().size();
        let ptr = out.id(self);

        let (operands, extra_args) = match view {
            Some(view) => (
                TensorAddressingOperands::TENSOR_VIEW,
                vec![Operand::IdRef(view)],
            ),
            None => (TensorAddressingOperands::NONE, vec![]),
        };

        self.cooperative_matrix_store_tensor_nv(
            ptr,
            mat_obj,
            layout,
            MemoryAccess::ALIGNED,
            [align.into()],
            operands,
            extra_args,
        )
        .unwrap();
    }

    fn compile_execute(
        &mut self,
        mat_a: core::Value,
        mat_b: core::Value,
        mat_c: core::Value,
        mat_d: core::Value,
    ) {
        let mat_a = self.compile_value(mat_a);
        let mat_b = self.compile_value(mat_b);
        let mat_c = self.compile_value(mat_c);
        let mat_d = self.compile_value(mat_d);

        let mat_a_id = self.read(&mat_a);
        let mat_b_id = self.read(&mat_b);
        let mat_c_id = self.read(&mat_c);
        let mat_d_id = self.write_id_cmma(&mat_d);

        let ty = mat_d.item().unwrap_ptr().id(self);

        let mut operands = CooperativeMatrixOperands::NONE_KHR;
        if matches!(mat_a.elem(), Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_A_SIGNED_COMPONENTS_KHR;
        }
        if matches!(mat_b.elem(), Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_B_SIGNED_COMPONENTS_KHR;
        }
        if matches!(mat_c.elem(), Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_C_SIGNED_COMPONENTS_KHR;
        }
        if matches!(mat_d.elem(), Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_RESULT_SIGNED_COMPONENTS_KHR;
        }

        self.cooperative_matrix_mul_add_khr(
            ty,
            Some(mat_d_id),
            mat_a_id,
            mat_b_id,
            mat_c_id,
            Some(operands),
        )
        .unwrap();

        self.write_cmma(&mat_d, mat_d_id);
    }

    fn compile_elementwise_op(&mut self, matrix: core::Value, op: Id, output: core::Value) {
        self.capabilities
            .insert(Capability::CooperativeMatrixPerElementOperationsNV);

        let matrix = self.compile_value(matrix);
        let output = self.compile_value(output);

        let matrix_ty = matrix.item().unwrap_ptr().id(self);
        let matrix_id = self.read(&matrix);
        let output_id = self.write_id_cmma(&output);

        let captures = self.opt.global_state.extra_functions[&op]
            .implicit_params
            .clone();
        let captures = captures
            .into_iter()
            .map(|var| self.compile_value(var))
            .collect::<Vec<_>>();
        let captures = captures
            .iter()
            .map(|var| self.read(var))
            .collect::<Vec<_>>();
        let func = self.state.extra_funcs[&op].id;

        self.cooperative_matrix_per_element_op_nv(
            matrix_ty,
            Some(output_id),
            matrix_id,
            func,
            captures,
        )
        .unwrap();

        self.write_cmma(&output, output_id);
    }

    fn compile_cast(&mut self, input: core::Value, output: core::Value) {
        let input = self.compile_value(input);
        let output = self.compile_value(output);

        let input_ident = matrix_ident(&input);
        let output_ident = matrix_ident(&output);

        if input_ident != output_ident {
            self.capabilities
                .insert(Capability::CooperativeMatrixConversionsNV);
        }

        let input_ty = input.item();
        let output_ty = output.item().unwrap_ptr();

        let output_ty_id = output_ty.id(self);

        let input_id = self.read(&input);
        let output_id = self.write_id_cmma(&output);

        if input_ty == output_ty && input_ident != output_ident {
            self.cooperative_matrix_convert_nv(output_ty_id, Some(output_id), input_id)
                .unwrap()
        } else {
            input_ty.cast_to(self, Some(output_id), input_id, &output_ty)
        };

        self.write_cmma(&output, output_id);
    }

    pub fn matrix_rows(&mut self, mat: &Matrix) -> u32 {
        let rows = match mat.ident {
            CooperativeMatrixUse::MatrixAKHR => mat.m,
            CooperativeMatrixUse::MatrixBKHR => mat.k,
            CooperativeMatrixUse::MatrixAccumulatorKHR => mat.m,
        };
        self.const_u32(rows)
    }

    pub fn matrix_columns(&mut self, mat: &Matrix) -> u32 {
        let columns = match mat.ident {
            CooperativeMatrixUse::MatrixAKHR => mat.k,
            CooperativeMatrixUse::MatrixBKHR => mat.n,
            CooperativeMatrixUse::MatrixAccumulatorKHR => mat.n,
        };
        self.const_u32(columns)
    }

    pub fn compile_matrix(&mut self, mat: &core::MatrixType) -> Matrix {
        let elem = self.compile_type(core::Type::new(mat.storage)).elem();
        let ident = match mat.ident {
            core::MatrixIdent::A => CooperativeMatrixUse::MatrixAKHR,
            core::MatrixIdent::B => CooperativeMatrixUse::MatrixBKHR,
            core::MatrixIdent::Accumulator => CooperativeMatrixUse::MatrixAccumulatorKHR,
        };
        let layout = compile_layout(mat.layout);
        let scope = compile_scope(mat.scope);

        Matrix {
            id: 0,
            ident,
            m: mat.m as u32,
            n: mat.n as u32,
            k: mat.k as u32,
            elem,
            layout,
            scope,
        }
    }
}

fn matrix_ident(var: &Value) -> CooperativeMatrixUse {
    let Item::CoopMatrix { ident, .. } = var.item().unwrap_ptr() else {
        unreachable!()
    };
    ident
}

fn matrix_layout(var: &Value) -> Option<CooperativeMatrixLayout> {
    let Item::CoopMatrix { layout, .. } = var.item().unwrap_ptr() else {
        unreachable!()
    };
    layout
}

fn compile_layout(layout: MatrixLayout) -> Option<CooperativeMatrixLayout> {
    match layout {
        core::MatrixLayout::ColMajor => Some(CooperativeMatrixLayout::ColumnMajorKHR),
        core::MatrixLayout::RowMajor => Some(CooperativeMatrixLayout::RowMajorKHR),
        core::MatrixLayout::Undefined => None,
    }
}

fn compile_scope(scope: MatrixScope) -> spirv::Scope {
    match scope {
        MatrixScope::Plane => spirv::Scope::Subgroup,
        MatrixScope::Cube => spirv::Scope::Workgroup,
    }
}
