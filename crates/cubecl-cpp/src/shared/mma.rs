use crate::shared::Item;

use super::{Component, Dialect, Elem, Value};
use cubecl_core::ir::{
    DeviceProperties,
    features::{MmaConfig, ScaledMmaConfig},
};
use std::{
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
};

pub type SupportedMmaCombinations = Vec<MmaConfig>;
pub type SupportedScaledMmaCombinations = Vec<ScaledMmaConfig>;

pub trait Architecture {
    fn warp_size(&self) -> u32;
    fn is_wmma_capable(&self) -> bool;
    fn is_mfma_capable(&self) -> bool;
    fn get_version(&self) -> u32 {
        0
    }
}

pub fn register_wmma_features(
    supported_combinations: SupportedMmaCombinations,
    properties: &mut DeviceProperties,
) {
    for config in supported_combinations {
        properties.features.matmul.cmma.insert(config);
    }
}

pub fn register_mma_features(
    supported_combinations: SupportedMmaCombinations,
    properties: &mut DeviceProperties,
) {
    for config in supported_combinations {
        properties.features.matmul.mma.insert(config);
    }
}

pub fn register_scaled_mma_features(
    supported_combinations: SupportedScaledMmaCombinations,
    properties: &mut DeviceProperties,
) {
    for config in supported_combinations {
        properties.features.matmul.scaled_mma.insert(config);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum FragmentIdent<D: Dialect> {
    A,
    B,
    Accumulator,
    _Dialect(PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum FragmentLayout<D: Dialect> {
    ColMajor,
    RowMajor,
    _Dialect(PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct FragmentType<D: Dialect> {
    pub ident: FragmentIdent<D>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub elem: Elem<D>,
    pub layout: Option<FragmentLayout<D>>,
}

#[derive(new, Debug, Clone, PartialEq, Eq, Copy)]
pub struct MmaShape<D: Dialect> {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    _d: PhantomData<D>,
}

impl<D: Dialect> MmaShape<D> {
    pub fn num_elems(&self, ident: FragmentIdent<D>) -> u32 {
        match ident {
            FragmentIdent::A => self.m * self.k,
            FragmentIdent::B => self.k * self.n,
            FragmentIdent::Accumulator => self.m * self.n,
            _ => unimplemented!(),
        }
    }
}

/// Warp Matrix-Multiply and Accumulate Instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum WmmaInstruction<D: Dialect> {
    /// Fill the fragment with the value.
    Fill { frag: Value<D>, value: Value<D> },
    /// Load the value into the fragment given the stride.
    Load {
        frag: Value<D>,
        ptr: Value<D>,
        stride: Value<D>,
        layout: Option<FragmentLayout<D>>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        frag_a: Value<D>,
        frag_b: Value<D>,
        frag_c: Value<D>,
        frag_d: Value<D>,
        warp_size: u32,
    },
    /// Executes D=A*B+C using manually managed registers;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    /// Takes a sequence of registers for the inputs, and returns an array of registers for the
    /// output. PTX requires output registers to be non-overlapping, so we use array to ensure that
    /// and handle potentially destructuring it internally.
    ExecuteManual {
        shape: MmaShape<D>,
        frag_a: Value<D>,
        frag_b: Value<D>,
        frag_c: Value<D>,
        frag_d: Value<D>,
    },
    /// Executes D=A*B+C using manually managed registers;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    /// Takes a sequence of registers for the inputs, and returns an array of registers for the
    /// output. PTX requires output registers to be non-overlapping, so we use array to ensure that
    /// and handle potentially destructuring it internally.
    ExecuteScaled {
        shape: MmaShape<D>,
        frag_a: Value<D>,
        frag_b: Value<D>,
        frag_c: Value<D>,
        frag_d: Value<D>,

        scales_a: Value<D>,
        scales_b: Value<D>,
        scales_factor: u32,
    },
    /// Store the fragment in an output variable following the stride and the layout.
    Store {
        frag: Value<D>,
        stride: Value<D>,
        destination: Value<D>,
        layout: FragmentLayout<D>,
    },
    /// Load a part of a fragment into registers, either 1, 2, or 4 at once.
    LdMatrix {
        output: Value<D>,
        ptr: Value<D>,
        factor: u32,
        transpose: bool,
    },
    /// Store a part of a fragment into smem, either 1, 2, or 4 at once.
    StMatrix {
        registers: Value<D>,
        ptr: Value<D>,
        factor: u32,
        transpose: bool,
    },
    /// Cast
    Cast { input: Value<D>, output: Value<D> },
}

impl<D: Dialect> Display for FragmentLayout<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_wmma_fragment_layout(f, self)
    }
}

impl<D: Dialect> Display for FragmentIdent<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_wwma_fragment_ident(f, self)
    }
}

impl<D: Dialect> Display for FragmentType<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_wmma_fragment(f, self)
    }
}

impl<D: Dialect> Display for WmmaInstruction<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_wmma_instruction(f, self)
    }
}

pub mod wmma_api_base {
    use crate::{
        cuda::ptx::{ldmatrix_call, stmatrix_call},
        shared::ManualMma,
    };

    use super::*;

    pub fn compile_fragment_declaration<D: Dialect>(
        f: &mut std::fmt::Formatter<'_>,
        val: &Value<D>,
        ty: &Item<D>,
    ) -> std::fmt::Result {
        match ty {
            Item::Fragment(frag) => writeln!(f, "{frag} {val}_store;"),
            _ => panic!("value must be a fragment"),
        }
    }

    pub fn compile_fragment_ident<D: Dialect>(
        f: &mut std::fmt::Formatter<'_>,
        namespace: &str,
        ident: &FragmentIdent<D>,
    ) -> std::fmt::Result {
        match ident {
            FragmentIdent::A => write!(f, "{namespace}::matrix_a"),
            FragmentIdent::B => write!(f, "{namespace}::matrix_b"),
            FragmentIdent::Accumulator => write!(f, "{namespace}::accumulator"),
            FragmentIdent::_Dialect(_) => Ok(()),
        }
    }

    pub fn compile_fragment_layout<D: Dialect>(
        f: &mut std::fmt::Formatter<'_>,
        namespace: &str,
        layout: &FragmentLayout<D>,
    ) -> std::fmt::Result {
        match layout {
            FragmentLayout::ColMajor => f.write_str(format!("{namespace}::col_major").as_str()),
            FragmentLayout::RowMajor => f.write_str(format!("{namespace}::row_major").as_str()),
            FragmentLayout::_Dialect(_) => Ok(()),
        }
    }

    pub fn compile_fragment<D: Dialect>(
        f: &mut std::fmt::Formatter<'_>,
        namespace: &str,
        fragment: &FragmentType<D>,
    ) -> std::fmt::Result {
        let elem = match fragment.elem {
            Elem::TF32 => format!("{namespace}::precision::tf32"),
            Elem::BF16 => {
                if fragment.ident == FragmentIdent::Accumulator {
                    format!("{}", Elem::<D>::F16) // Normally not supported except for cast.
                } else {
                    format!("{}", fragment.elem)
                }
            }
            elem => format!("{elem}"),
        };
        match fragment.layout {
            Some(layout) => write!(
                f,
                "{namespace}::fragment<{}, {}, {}, {}, {}, {}>",
                fragment.ident, fragment.m, fragment.n, fragment.k, elem, layout
            ),
            None => write!(
                f,
                "{namespace}::fragment<{}, {}, {}, {}, {}>",
                fragment.ident, fragment.m, fragment.n, fragment.k, elem,
            ),
        }
    }

    pub fn compile_instruction<D: Dialect>(
        f: &mut std::fmt::Formatter<'_>,
        namespace: &str,
        instruction: &WmmaInstruction<D>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag, value } => {
                let frag = frag.fmt_ref();
                writeln!(f, "{namespace}::fill_fragment({frag}, {value});")
            }
            WmmaInstruction::Load {
                frag,
                ptr,
                stride,
                layout: None,
            } => {
                let item = *ptr.item().value_ty();
                let frag = frag.fmt_ref();
                if item.vectorization() > 1 {
                    let elem = item.elem();
                    let qualifier = ptr.const_qualifier();
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, reinterpret_cast<{elem}{qualifier}*>({ptr}), {stride});"
                    )
                } else {
                    writeln!(f, "{namespace}::load_matrix_sync({frag}, {ptr}, {stride});")
                }
            }
            WmmaInstruction::Load {
                frag,
                ptr,
                stride,
                layout: Some(layout),
            } => {
                let frag = frag.fmt_ref();
                let layout = match layout {
                    FragmentLayout::ColMajor => format!("{namespace}::mem_col_major"),
                    FragmentLayout::RowMajor => format!("{namespace}::mem_row_major"),
                    FragmentLayout::_Dialect(_) => "".to_string(),
                };
                let item = *ptr.item().value_ty();
                if item.vectorization() > 1 {
                    let elem = item.elem();
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, reinterpret_cast<{elem} *>({ptr}), {stride}, {layout});"
                    )
                } else {
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, {ptr}, {stride}, {layout});"
                    )
                }
            }
            WmmaInstruction::LdMatrix {
                output,
                ptr,
                factor,
                transpose,
            } => f.write_str(&ldmatrix_call(output, ptr, factor, transpose)),
            WmmaInstruction::StMatrix {
                registers,
                ptr,
                factor,
                transpose,
            } => f.write_str(&stmatrix_call(registers, ptr, factor, transpose)),
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                ..
            } => {
                let frag_a = frag_a.fmt_ref();
                let frag_b = frag_b.fmt_ref();
                let frag_c = frag_c.fmt_ref();
                let frag_d = frag_d.fmt_ref();
                writeln!(
                    f,
                    "{namespace}::mma_sync({frag_d}, {frag_a}, {frag_b}, {frag_c});"
                )
            }
            WmmaInstruction::Store {
                frag,
                stride,
                destination,
                layout,
            } => {
                let frag = frag.fmt_ref();
                let layout = match layout {
                    FragmentLayout::ColMajor => format!("{namespace}::mem_col_major"),
                    FragmentLayout::RowMajor => format!("{namespace}::mem_row_major"),
                    FragmentLayout::_Dialect(_) => "".to_string(),
                };

                let item = *destination.item().value_ty();
                let mut reinterpret_cast = item.vectorization() > 1;
                let elem = match item.elem() {
                    Elem::BF16 => {
                        reinterpret_cast = true;
                        Elem::F16
                    }
                    _ => *item.elem(),
                };
                if reinterpret_cast {
                    writeln!(
                        f,
                        "{namespace}::store_matrix_sync(reinterpret_cast<{elem} *>({destination}), {frag}, {stride}, {layout});"
                    )
                } else {
                    writeln!(
                        f,
                        "{namespace}::store_matrix_sync({destination}, {frag}, {stride}, {layout});"
                    )
                }
            }
            WmmaInstruction::Cast { input, output } => {
                let input = input.ensure_lvalue(f)?;
                let output = output.ensure_lvalue(f)?;
                let ty = match *output.item().value_ty() {
                    Item::Fragment(frag) => frag.elem,
                    _ => panic!("Should be a fragment"),
                };
                match ty {
                    Elem::BF16 => {
                        let elem = Elem::<D>::F16;
                        write!(
                            f,
                            "// cast
for(int t=0; t<{input}.num_elements; t++) {{
  {ty} elem = {ty}({input}.x[t]);
  {output}.x[t] = *reinterpret_cast<{elem} *>(&elem);
}}
"
                        )
                    }
                    _ => {
                        write!(
                            f,
                            "// cast
for(int t=0; t<{input}.num_elements; t++) {{ {output}.x[t] = {ty}({input}.x[t]); }}
"
                        )
                    }
                }
            }
            WmmaInstruction::ExecuteManual {
                shape,
                frag_a,
                frag_b,
                frag_c,
                frag_d,
            } => D::compile_manual_mma(f, ManualMma::new(*shape, frag_a, frag_b, frag_c, frag_d)),
            WmmaInstruction::ExecuteScaled {
                shape,
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                scales_a,
                scales_b,
                scales_factor,
            } => D::compile_scaled_mma(
                f,
                ManualMma::new(*shape, frag_a, frag_b, frag_c, frag_d),
                *scales_a,
                *scales_b,
                *scales_factor,
            ),
        }
    }
}

pub fn frag_as_ptr<D: Dialect>(f: &mut Formatter<'_>, ptr: &Value<D>) -> Value<D> {
    let item = ptr.item();
    if item.vectorization() > 1 {
        let item_value = item.as_scalar();
        ptr.reinterpret_ptr(f, item_value)
    } else {
        *ptr
    }
}

pub fn frag_ident_str<D: Dialect>(frag: &FragmentIdent<D>) -> &str {
    match frag {
        FragmentIdent::A => "a",
        FragmentIdent::B => "b",
        FragmentIdent::Accumulator => "c",
        FragmentIdent::_Dialect(_) => "d",
    }
}

pub fn frag_layout_str<D: Dialect>(frag: &Option<FragmentLayout<D>>) -> &str {
    match frag {
        Some(layout) => match layout {
            FragmentLayout::ColMajor => "col",
            FragmentLayout::RowMajor => "row",
            FragmentLayout::_Dialect(_) => "",
        },
        None => "",
    }
}

pub fn value_to_frag<D: Dialect>(frag: &Value<D>) -> FragmentType<D> {
    match frag.item().unwrap_ptr() {
        Item::Fragment(frag) => frag,
        _ => panic!(),
    }
}
