use crate::{
    item::{Elem, Item},
    lookups::Matrix,
    variable::{ConstVal, IndexedVariable, Variable},
    SpirvCompiler, SpirvTarget,
};
use cubecl_core::ir::{self as core, CoopMma};
use rspirv::spirv::{CooperativeMatrixLayout, CooperativeMatrixUse, Word};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_cmma(&mut self, cmma: CoopMma) {
        match cmma {
            CoopMma::Fill { mat, value } => self.compile_fill(mat, value),
            CoopMma::Load { mat, value, stride } => self.compile_load(mat, value, stride),
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => self.compile_execute(mat_a, mat_b, mat_c, mat_d),
            CoopMma::Store {
                output,
                mat,
                stride,
                ..
            } => self.compile_store(mat, output, stride),
        }
    }

    fn compile_load(&mut self, mat: core::Variable, value: core::Variable, stride: core::Variable) {
        let mat = self.compile_variable(mat);
        let (id, depth, mat) = self.matrix_var(&mat);

        let value = self.compile_variable(value);
        let stride = self.compile_variable(stride);
        let stride = self.read(&stride);
        let layout = mat.layout.unwrap_or(CooperativeMatrixLayout::RowMajorKHR);
        let memory_layout = self.const_u32(layout as u32);
        let ptr = self.index_ptr(&value);

        let out_ty = self.item(&mat);
        let ty = out_ty.id(self);

        let mat_id = self
            .cooperative_matrix_load_khr(ty, None, ptr, memory_layout, Some(stride), None, vec![])
            .unwrap();
        self.state.matrices.get_mut(&(id, depth)).unwrap().id = mat_id;
    }

    fn compile_fill(&mut self, mat: core::Variable, value: core::Variable) {
        let mat = self.compile_variable(mat);
        let value = self.compile_variable(value);
        let (id, depth, mat) = self.matrix_var(&mat);
        let item = self.item(&mat);
        let ty = item.id(self);
        let mat_id = match value {
            Variable::ConstantScalar(id, value, _) => {
                let const_mat =
                    self.get_or_insert_const(value, item, |b| b.constant_composite(ty, vec![id]));
                self.copy_object(ty, None, const_mat).unwrap()
            }
            var => {
                let var = self.read(&var);
                self.composite_construct(ty, None, vec![var]).unwrap()
            }
        };

        self.state.matrices.get_mut(&(id, depth)).unwrap().id = mat_id;
    }

    fn compile_store(&mut self, mat: core::Variable, out: core::Variable, stride: core::Variable) {
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).2;
        let mat_obj = mat.id;
        assert_ne!(mat_obj, 0, "Can't store uninitialized matrix");

        let out = self.compile_variable(out);
        let stride = self.compile_variable(stride);
        let stride = self.read(&stride);
        let layout = mat.layout.unwrap_or(CooperativeMatrixLayout::RowMajorKHR);
        let memory_layout = self.const_u32(layout as u32);
        let ptr = self.index_ptr(&out);

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

        let mat_a = self.matrix_var(&mat_a).2;
        let mat_b = self.matrix_var(&mat_b).2;
        let mat_c = self.matrix_var(&mat_c).2;
        let (d_id, d_depth, mat_d) = self.matrix_var(&mat_d);

        assert_ne!(mat_a.id, 0, "Can't execute with uninitialized matrix");
        assert_ne!(mat_b.id, 0, "Can't execute with uninitialized matrix");
        assert_ne!(mat_c.id, 0, "Can't execute with uninitialized matrix");
        let ty = self.item(&mat_d).id(self);

        let mat_d_id = self
            .cooperative_matrix_mul_add_khr(ty, None, mat_a.id, mat_b.id, mat_c.id, None)
            .unwrap();

        self.state.matrices.get_mut(&(d_id, d_depth)).unwrap().id = mat_d_id;
    }

    fn matrix_var(&mut self, var: &Variable) -> (u16, u8, Matrix) {
        let (id, depth) = match var {
            Variable::CoopMatrix(id, depth) => (*id, *depth),
            _ => unreachable!(),
        };
        let mat = self.state.matrices[&(id, depth)];
        (id, depth, mat)
    }

    fn index_ptr(&mut self, var: &Variable) -> Word {
        let zero = self.const_u32(0);
        let zero = Variable::ConstantScalar(zero, ConstVal::Bit32(0), Elem::Int(32, false));
        match self.index(var, &zero, false) {
            IndexedVariable::Pointer(ptr, _) => ptr,
            _ => unreachable!("CMMA store always takes array pointer"),
        }
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

    pub fn init_coop_matrix(&mut self, mat: core::Matrix) -> Matrix {
        let elem = self.compile_item(core::Item::new(mat.elem)).elem();
        let ident = match mat.ident {
            core::MatrixIdent::A => CooperativeMatrixUse::MatrixAKHR,
            core::MatrixIdent::B => CooperativeMatrixUse::MatrixBKHR,
            core::MatrixIdent::Accumulator => CooperativeMatrixUse::MatrixAccumulatorKHR,
        };
        let layout = match mat.layout {
            core::MatrixLayout::ColMajor => Some(CooperativeMatrixLayout::ColumnMajorKHR),
            core::MatrixLayout::RowMajor => Some(CooperativeMatrixLayout::RowMajorKHR),
            core::MatrixLayout::Undefined => None,
        };

        Matrix {
            id: 0,
            ident,
            m: mat.m,
            n: mat.n,
            k: mat.k,
            elem,
            layout,
        }
    }
}
