use core::fmt::Formatter;

use crate::{
    Dialect,
    cuda::ptx,
    shared::{Elem, FragmentIdent, MmaShape},
};

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension<D: Dialect> {
    #[default]
    NoExtension,
    Mma(MmaExtension<D>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MmaExtension<D: Dialect> {
    Execute(MmaExecute<D>),
    ExecuteScaled(MmaExecuteScaled<D>),
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Fragment<D: Dialect>(pub Elem<D>);

impl<D: Dialect> Fragment<D> {
    pub fn elem(&self) -> Elem<D> {
        self.0
    }
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaExecute<D: Dialect> {
    pub shape: MmaShape<D>,
    pub frag_a: Fragment<D>,
    pub frag_b: Fragment<D>,
    pub frag_c: Fragment<D>,
    pub frag_d: Fragment<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaExecuteScaled<D: Dialect> {
    pub shape: MmaShape<D>,
    pub frag_a: Fragment<D>,
    pub frag_b: Fragment<D>,
    pub frag_c: Fragment<D>,
    pub frag_d: Fragment<D>,
    pub scales_elem: Elem<D>,
    pub scales_factor: u32,
}

impl<D: Dialect> MmaExtension<D> {
    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MmaExtension::Execute(mma_execute) => mma_execute.format_extension(f),
            MmaExtension::ExecuteScaled(mma_execute) => mma_execute.format_extension(f),
        }
    }
}

impl<D: Dialect> MmaExecute<D> {
    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let a_elems = self.shape.num_elems(FragmentIdent::<D>::A) / 32;
        let b_elems = self.shape.num_elems(FragmentIdent::<D>::B) / 32;
        let c_elems = self.shape.num_elems(FragmentIdent::<D>::Accumulator) / 32;
        let d_elems = self.shape.num_elems(FragmentIdent::<D>::Accumulator) / 32;

        let a_regs = a_elems as usize / (32 / self.frag_a.elem().unpacked().size_bits());
        let b_regs = b_elems as usize / (32 / self.frag_b.elem().unpacked().size_bits());
        let c_regs = c_elems as usize / (32 / self.frag_c.elem().unpacked().size_bits());
        let d_regs = d_elems as usize / (32 / self.frag_d.elem().unpacked().size_bits());

        let ptx = ptx::mma_template(
            self.frag_a.elem().unpacked(),
            self.frag_b.elem().unpacked(),
            self.frag_c.elem().unpacked(),
            self.shape.k,
            a_regs,
            b_regs,
            c_regs,
            d_regs,
        );

        f.write_str(&ptx)
    }
}

impl<D: Dialect> MmaExecuteScaled<D> {
    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let a_elems = self.shape.num_elems(FragmentIdent::<D>::A) / 32;
        let b_elems = self.shape.num_elems(FragmentIdent::<D>::B) / 32;
        let c_elems = self.shape.num_elems(FragmentIdent::<D>::Accumulator) / 32;
        let d_elems = self.shape.num_elems(FragmentIdent::<D>::Accumulator) / 32;

        let a_regs = a_elems as usize / (32 / self.frag_a.elem().unpacked().size_bits());
        let b_regs = b_elems as usize / (32 / self.frag_b.elem().unpacked().size_bits());
        let c_regs = c_elems as usize / (32 / self.frag_c.elem().unpacked().size_bits());
        let d_regs = d_elems as usize / (32 / self.frag_d.elem().unpacked().size_bits());

        let ptx = ptx::mma_scaled_template(
            self.frag_a.elem().unpacked(),
            self.frag_b.elem().unpacked(),
            self.frag_c.elem().unpacked(),
            self.shape.k,
            a_regs,
            b_regs,
            c_regs,
            d_regs,
            self.scales_elem,
            self.scales_factor,
        );

        f.write_str(&ptx)
    }
}
