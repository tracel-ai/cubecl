use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    lookups::Matrix,
    variable::Variable,
};
use cubecl_core::ir::{self as core, CoopMma, Id, MatrixLayout};
use rspirv::spirv::{
    Capability, CooperativeMatrixLayout, CooperativeMatrixOperands, CooperativeMatrixUse,
    StorageClass,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_cmma(&mut self, cmma: CoopMma, out: Option<core::Variable>) {
        self.capabilities.insert(Capability::CooperativeMatrixKHR);
        let out = out.unwrap();
        match cmma {
            CoopMma::Fill { value } => self.compile_fill(out, value),
            CoopMma::Load {
                value,
                stride,
                layout,
                offset,
            } => self.compile_load(out, value, stride, offset, layout),
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => self.compile_execute(mat_a, mat_b, mat_c, out),
            CoopMma::Store {
                mat,
                stride,
                layout,
                offset,
            } => self.compile_store(mat, out, stride, offset, layout),
            CoopMma::Cast { input } => self.compile_cast(input, out),
        }
    }

    fn compile_load(
        &mut self,
        mat: core::Variable,
        value: core::Variable,
        stride: core::Variable,
        offset: core::Variable,
        layout: Option<MatrixLayout>,
    ) {
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;

        let value = self.compile_variable(value);
        let stride = self.compile_variable(stride);
        let stride_item = stride.item();
        let mut stride = self.read(&stride);

        if let Item::Vector(_, line_size) = value.item() {
            let shift = stride_item.const_u32(self, line_size.trailing_zeros());
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

        let offset = self.compile_variable(offset);
        let ptr = self.index_ptr(&value, &offset);
        let out_ty = self.item(&mat);
        let ty = out_ty.id(self);

        let mat_id = self
            .cooperative_matrix_load_khr(ty, None, ptr, memory_layout, Some(stride), None, vec![])
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
            Variable::ConstantScalar(id, value, _) => {
                self.get_or_insert_const(value, item, |b| b.constant_composite(ty, vec![id]))
            }
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
        out: core::Variable,
        stride: core::Variable,
        offset: core::Variable,
        layout: MatrixLayout,
    ) {
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;
        let item = self.item(&mat);
        let ty = item.id(self);
        let mat_obj = self.load(ty, None, mat.id, None, vec![]).unwrap();
        //assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

        let out = self.compile_variable(out);
        let stride = self.compile_variable(stride);
        let stride_item = stride.item();
        let mut stride = self.read(&stride);
        let layout = compile_layout(layout).unwrap_or(CooperativeMatrixLayout::RowMajorKHR);
        let memory_layout = self.const_u32(layout as u32);
        let offset = self.compile_variable(offset);
        let ptr = self.index_ptr(&out, &offset);

        if let Item::Vector(_, line_size) = out.item() {
            let shift = stride_item.const_u32(self, line_size.trailing_zeros());
            let stride_ty = stride_item.id(self);
            stride = self
                .shift_right_logical(stride_ty, None, stride, shift)
                .unwrap();
        }

        self.cooperative_matrix_store_khr(ptr, mat_obj, memory_layout, Some(stride), None, vec![])
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

    fn compile_cast(&mut self, input: core::Variable, output: core::Variable) {
        let input = self.compile_variable(input);
        let output = self.compile_variable(output);

        let input = self.matrix_var(&input).1;
        let output = self.matrix_var(&output).1;

        let input_ty = self.item(&input).id(self);
        let output_ty = self.item(&output).id(self);

        let fragment_id = self.load(input_ty, None, input.id, None, vec![]).unwrap();

        let frag_new = self.f_convert(output_ty, None, fragment_id).unwrap();

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
        } as u32;
        self.const_u32(rows)
    }

    fn columns(&mut self, mat: &Matrix) -> u32 {
        let columns = match mat.ident {
            CooperativeMatrixUse::MatrixAKHR => mat.k,
            CooperativeMatrixUse::MatrixBKHR => mat.n,
            CooperativeMatrixUse::MatrixAccumulatorKHR => mat.n,
        } as u32;
        self.const_u32(columns)
    }

    pub fn item(&mut self, mat: &Matrix) -> Item {
        Item::CoopMatrix {
            ty: mat.elem,
            rows: self.rows(mat),
            columns: self.columns(mat),
            ident: mat.ident,
        }
    }

    pub fn init_coop_matrix(&mut self, mat: core::Matrix, var: core::Variable) -> Matrix {
        let elem = self.compile_item(core::Item::new(mat.elem)).elem();
        let ident = match mat.ident {
            core::MatrixIdent::A => CooperativeMatrixUse::MatrixAKHR,
            core::MatrixIdent::B => CooperativeMatrixUse::MatrixBKHR,
            core::MatrixIdent::Accumulator => CooperativeMatrixUse::MatrixAccumulatorKHR,
        };
        let layout = compile_layout(mat.layout);

        let mut mat = Matrix {
            id: 0,
            ident,
            m: mat.m,
            n: mat.n,
            k: mat.k,
            elem,
            layout,
        };

        let item = Item::Pointer(StorageClass::Function, Box::new(self.item(&mat)));
        let ty = item.id(self);
        mat.id = self.declare_function_variable(ty);
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
