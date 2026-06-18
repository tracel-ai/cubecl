use cubecl_core::cmma::{MatrixIdent, MatrixShape};
use pliron::{context::Context, r#type::TypeHandle};

use crate::{cuda::ptx, shared::ty::TypedExtCPP};

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension {
    #[default]
    NoExtension,
    Mma(MmaExtension),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MmaExtension {
    Execute(MmaExecute),
    ExecuteScaled(MmaExecuteScaled),
    LdMatrix(LdMatrix),
    StMatrix(StMatrix),
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaExecute {
    pub shape: MatrixShape,
    pub elem_a: TypeHandle,
    pub elem_b: TypeHandle,
    pub elem_cd: TypeHandle,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaExecuteScaled {
    pub shape: MatrixShape,
    pub elem_a: TypeHandle,
    pub elem_b: TypeHandle,
    pub elem_cd: TypeHandle,
    pub scales_elem: TypeHandle,
    pub scales_factor: usize,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct LdMatrix {
    pub elem: TypeHandle,
    pub factor: usize,
    pub transpose: bool,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct StMatrix {
    pub elem: TypeHandle,
    pub factor: usize,
    pub transpose: bool,
}

impl MmaExtension {
    pub fn format_extension(&self, ctx: &Context) -> String {
        match self {
            MmaExtension::Execute(mma_execute) => mma_execute.format_extension(ctx),
            MmaExtension::ExecuteScaled(mma_execute) => mma_execute.format_extension(ctx),
            MmaExtension::LdMatrix(ld_matrix) => ld_matrix.format_extension(ctx),
            MmaExtension::StMatrix(st_matrix) => st_matrix.format_extension(ctx),
        }
    }
}

impl MmaExecute {
    pub fn format_extension(&self, ctx: &Context) -> String {
        let a_elems = self.shape.num_elems(MatrixIdent::A) / 32;
        let b_elems = self.shape.num_elems(MatrixIdent::B) / 32;
        let c_elems = self.shape.num_elems(MatrixIdent::Accumulator) / 32;
        let d_elems = self.shape.num_elems(MatrixIdent::Accumulator) / 32;

        let a_regs = a_elems / (32 / self.elem_a.unpacked_size_bits(ctx));
        let b_regs = b_elems / (32 / self.elem_b.unpacked_size_bits(ctx));
        let c_regs = c_elems / (32 / self.elem_cd.unpacked_size_bits(ctx));
        let d_regs = d_elems / (32 / self.elem_cd.unpacked_size_bits(ctx));

        ptx::mma_template(
            ctx,
            self.elem_a,
            self.elem_b,
            self.elem_cd,
            self.shape.k,
            a_regs,
            b_regs,
            c_regs,
            d_regs,
        )
    }
}

impl MmaExecuteScaled {
    pub fn format_extension(&self, ctx: &Context) -> String {
        let a_elems = self.shape.num_elems(MatrixIdent::A) / 32;
        let b_elems = self.shape.num_elems(MatrixIdent::B) / 32;
        let c_elems = self.shape.num_elems(MatrixIdent::Accumulator) / 32;
        let d_elems = self.shape.num_elems(MatrixIdent::Accumulator) / 32;

        let a_regs = a_elems / (32 / self.elem_a.unpacked_size_bits(ctx));
        let b_regs = b_elems / (32 / self.elem_b.unpacked_size_bits(ctx));
        let c_regs = c_elems / (32 / self.elem_cd.unpacked_size_bits(ctx));
        let d_regs = d_elems / (32 / self.elem_cd.unpacked_size_bits(ctx));

        ptx::mma_scaled_template(
            ctx,
            self.elem_a,
            self.elem_b,
            self.elem_cd,
            self.shape.k,
            a_regs,
            b_regs,
            c_regs,
            d_regs,
            self.scales_elem,
            self.scales_factor,
        )
    }
}

impl LdMatrix {
    pub fn format_extension(&self, ctx: &Context) -> String {
        ptx::ldmatrix_template(ctx, self.elem, self.factor, self.transpose)
    }
}

impl StMatrix {
    pub fn format_extension(&self, ctx: &Context) -> String {
        ptx::stmatrix_template(ctx, self.elem, self.factor, self.transpose)
    }
}
