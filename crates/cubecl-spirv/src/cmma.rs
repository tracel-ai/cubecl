use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    lookups::Matrix,
    variable::Variable,
};
use cubecl_core::ir::{self as core, CoopMma, ElemType, Id, MatrixLayout, MatrixScope};
use rspirv::{
    dr::Operand,
    spirv::{
        self, Capability, CooperativeMatrixLayout, CooperativeMatrixOperands, CooperativeMatrixUse,
        MemoryAccess, StorageClass, TensorAddressingOperands,
    },
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_cmma(&mut self, cmma: CoopMma, out: Option<core::Variable>) {
        self.capabilities.insert(Capability::CooperativeMatrixKHR);

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
        mat: core::Variable,
        ptr: core::Variable,
        stride: core::Variable,
        layout: Option<MatrixLayout>,
    ) {
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;

        let ptr = self.compile_variable(ptr);
        let stride = self.compile_variable(stride);
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
            .or(mat.layout)
            .unwrap_or(CooperativeMatrixLayout::RowMajorKHR);
        let memory_layout = self.const_u32(layout as u32);

        let ptr = self.read(&ptr);
        let out_ty = self.item(&mat);
        let ty = out_ty.id(self);

        let mat_id = self
            .cooperative_matrix_load_khr(
                ty,
                None,
                ptr,
                memory_layout,
                Some(stride),
                Some(MemoryAccess::ALIGNED),
                [align.into()],
            )
            .unwrap();

        self.store(mat.id, mat_id, None, vec![]).unwrap();
    }

    fn compile_load_tensor(
        &mut self,
        mat: core::Variable,
        buffer: core::Variable,
        layout: core::Variable,
        view: Option<core::Variable>,
    ) {
        self.capabilities
            .insert(Capability::CooperativeMatrixTensorAddressingNV);

        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;

        let buffer = self.compile_variable(buffer);
        let layout = self.compile_variable(layout);
        let view = view.map(|view| self.compile_variable(view));
        let layout = self.read(&layout);
        let view = view.map(|view| self.read(&view));

        let ptr = buffer.id(self);
        let out_ty = self.item(&mat);
        let align = buffer.item().size();
        let ty = out_ty.id(self);

        let zero = Item::Scalar(mat.elem).const_u32(self, 0);
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
                None,
                ptr,
                clipped_fallback,
                layout,
                MemoryAccess::ALIGNED,
                [align.into()],
                operands,
                extra_args,
            )
            .unwrap();

        self.store(mat.id, mat_id, None, vec![]).unwrap();
    }

    fn compile_fill(&mut self, mat: core::Variable, value: core::Variable) {
        let mat = self.compile_variable(mat);
        let value = self.compile_variable(value);
        let mat = self.matrix_var(&mat).1;
        let item = self.item(&mat);
        let ty = item.id(self);
        let mat_id = match value {
            Variable::Constant(id, _, _) => self.constant_composite(ty, vec![id]),
            var => {
                let var = self.read(&var);
                self.composite_construct(ty, None, vec![var]).unwrap()
            }
        };

        self.store(mat.id, mat_id, None, vec![]).unwrap();
    }

    fn compile_store(
        &mut self,
        mat: core::Variable,
        stride: core::Variable,
        destination: core::Variable,
        layout: MatrixLayout,
    ) {
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;
        let item = self.item(&mat);
        let ty = item.id(self);
        let mat_obj = self.load(ty, None, mat.id, None, vec![]).unwrap();
        //assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

        let ptr = self.compile_variable(destination);
        let stride = self.compile_variable(stride);
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
        mat: core::Variable,
        out: core::Variable,
        layout: core::Variable,
        view: Option<core::Variable>,
    ) {
        self.capabilities
            .insert(Capability::CooperativeMatrixTensorAddressingNV);

        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;
        let item = self.item(&mat);
        let ty = item.id(self);
        let mat_obj = self.load(ty, None, mat.id, None, vec![]).unwrap();
        //assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

        let out = self.compile_variable(out);
        let layout = self.compile_variable(layout);
        let view = view.map(|view| self.compile_variable(view));

        let layout = self.read(&layout);
        let view = view.map(|view| self.read(&view));

        let align = out.item().size();
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
        mat_a: core::Variable,
        mat_b: core::Variable,
        mat_c: core::Variable,
        mat_d: core::Variable,
    ) {
        let mat_a = self.compile_variable(mat_a);
        let mat_b = self.compile_variable(mat_b);
        let mat_c = self.compile_variable(mat_c);
        let mat_d = self.compile_variable(mat_d);

        let mat_a = self.matrix_var(&mat_a).1;
        let mat_b = self.matrix_var(&mat_b).1;
        let mat_c = self.matrix_var(&mat_c).1;
        let mat_d = self.matrix_var(&mat_d).1;

        let mat_a_ty = self.item(&mat_a).id(self);
        let mat_b_ty = self.item(&mat_b).id(self);
        let mat_c_ty = self.item(&mat_c).id(self);

        let mat_a_id = self.load(mat_a_ty, None, mat_a.id, None, vec![]).unwrap();
        let mat_b_id = self.load(mat_b_ty, None, mat_b.id, None, vec![]).unwrap();
        let mat_c_id = self.load(mat_c_ty, None, mat_c.id, None, vec![]).unwrap();

        let ty = self.item(&mat_d).id(self);

        let mut operands = CooperativeMatrixOperands::NONE_KHR;
        if matches!(mat_a.elem, Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_A_SIGNED_COMPONENTS_KHR;
        }
        if matches!(mat_b.elem, Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_B_SIGNED_COMPONENTS_KHR;
        }
        if matches!(mat_c.elem, Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_C_SIGNED_COMPONENTS_KHR;
        }
        if matches!(mat_d.elem, Elem::Int(_, true)) {
            operands |= CooperativeMatrixOperands::MATRIX_RESULT_SIGNED_COMPONENTS_KHR;
        }

        let mat_d_id = self
            .cooperative_matrix_mul_add_khr(ty, None, mat_a_id, mat_b_id, mat_c_id, Some(operands))
            .unwrap();

        self.store(mat_d.id, mat_d_id, None, vec![]).unwrap();
    }

    fn compile_elementwise_op(&mut self, matrix: core::Variable, op: Id, output: core::Variable) {
        self.capabilities
            .insert(Capability::CooperativeMatrixPerElementOperationsNV);

        let matrix = self.compile_variable(matrix);
        let output = self.compile_variable(output);
        let matrix = self.matrix_var(&matrix).1;
        let output = self.matrix_var(&output).1;

        let matrix_ty = self.item(&matrix).id(self);
        let matrix_id = self.load(matrix_ty, None, matrix.id, None, vec![]).unwrap();

        let captures = self.opt.global_state.extra_functions[&op]
            .implicit_params
            .clone();
        let captures = captures
            .into_iter()
            .map(|var| self.compile_variable(var))
            .collect::<Vec<_>>();
        let captures = captures
            .iter()
            .map(|var| self.read(var))
            .collect::<Vec<_>>();
        let func = self.state.extra_funcs[&op].id;

        let mat_out_id = self
            .cooperative_matrix_per_element_op_nv(matrix_ty, None, matrix_id, func, captures)
            .unwrap();

        self.store(output.id, mat_out_id, None, vec![]).unwrap();
    }

    fn compile_cast(&mut self, input: core::Variable, output: core::Variable) {
        let input = self.compile_variable(input);
        let output = self.compile_variable(output);

        let input = self.matrix_var(&input).1;
        let output = self.matrix_var(&output).1;

        if input.ident != output.ident {
            self.capabilities
                .insert(Capability::CooperativeMatrixConversionsNV);
        }

        let input_ty = self.item(&input);
        let output_ty = self.item(&output);

        let input_ty_id = input_ty.id(self);
        let output_ty_id = output_ty.id(self);

        let fragment_id = self
            .load(input_ty_id, None, input.id, None, vec![])
            .unwrap();

        let frag_new = if input_ty == output_ty && input.ident != output.ident {
            self.cooperative_matrix_convert_nv(output_ty_id, None, fragment_id)
                .unwrap()
        } else {
            input_ty.cast_to(self, None, fragment_id, &output_ty)
        };

        self.store(output.id, frag_new, None, vec![]).unwrap();
    }

    fn matrix_var(&mut self, var: &Variable) -> (Id, Matrix) {
        let id = match var {
            Variable::CoopMatrix(id, _) => *id,
            _ => unreachable!(),
        };
        let mat = self.state.matrices[&id];
        (id, mat)
    }

    fn rows(&mut self, mat: &Matrix) -> u32 {
        let rows = match mat.ident {
            CooperativeMatrixUse::MatrixAKHR => mat.m,
            CooperativeMatrixUse::MatrixBKHR => mat.k,
            CooperativeMatrixUse::MatrixAccumulatorKHR => mat.m,
        };
        self.const_u32(rows)
    }

    fn columns(&mut self, mat: &Matrix) -> u32 {
        let columns = match mat.ident {
            CooperativeMatrixUse::MatrixAKHR => mat.k,
            CooperativeMatrixUse::MatrixBKHR => mat.n,
            CooperativeMatrixUse::MatrixAccumulatorKHR => mat.n,
        };
        self.const_u32(columns)
    }

    pub fn item(&mut self, mat: &Matrix) -> Item {
        Item::CoopMatrix {
            ty: mat.elem,
            rows: self.rows(mat),
            columns: self.columns(mat),
            ident: mat.ident,
            scope: mat.scope,
        }
    }

    pub fn compile_matrix(&mut self, mat: &core::Matrix) -> Matrix {
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

    pub fn init_coop_matrix(
        &mut self,
        mat: core::Matrix,
        var: core::Variable,
        init: Option<Id>,
    ) -> Matrix {
        if mat.storage.elem_type() == ElemType::Float(core::FloatKind::BF16) {
            self.capabilities
                .insert(Capability::BFloat16CooperativeMatrixKHR);
        }
        if matches!(
            mat.storage.elem_type(),
            ElemType::Float(core::FloatKind::E5M2 | core::FloatKind::E4M3)
        ) {
            self.capabilities
                .insert(Capability::Float8CooperativeMatrixEXT);
        }

        let mut mat = self.compile_matrix(&mat);

        let item = Item::Pointer(StorageClass::Function, Box::new(self.item(&mat)));
        let ty = item.id(self);
        mat.id = self.declare_function_variable(ty, init);
        self.debug_var_name(mat.id, var);

        mat
    }
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
