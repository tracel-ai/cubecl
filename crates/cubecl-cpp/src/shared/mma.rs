use cubecl_core::Feature;
use cubecl_core::ir::{self as gpu};
use cubecl_runtime::DeviceProperties;
use std::fmt::{Display, Formatter};
use std::{fmt::Debug, marker::PhantomData};

use super::{Component, Dialect, Elem, FmtLeft, Variable};

pub type SupportedWmmaCombinations = Vec<(gpu::Elem, gpu::Elem, gpu::Elem, Vec<(u8, u8, u8)>)>;

pub trait Architecture {
    fn warp_size(&self) -> u32;
    fn is_wmma_capable(&self) -> bool;
    fn is_mfma_capable(&self) -> bool;
    fn get_version(&self) -> u32 {
        0
    }
}

pub fn register_wmma_features(
    supported_combinations: SupportedWmmaCombinations,
    properties: &mut DeviceProperties<Feature>,
) {
    for (i, o, c, tdims) in supported_combinations {
        for (m, n, k) in tdims {
            properties.register_feature(Feature::Cmma {
                a: i,
                b: o,
                c,
                m,
                n,
                k,
            });
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FragmentIdent<D: Dialect> {
    A,
    B,
    Accumulator,
    _Dialect(PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FragmentLayout<D: Dialect> {
    ColMajor,
    RowMajor,
    _Dialect(PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Fragment<D: Dialect> {
    pub ident: FragmentIdent<D>,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub elem: Elem<D>,
    pub layout: Option<FragmentLayout<D>>,
}

/// Warp Matrix-Multiply and Accumulate Instruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WmmaInstruction<D: Dialect> {
    /// Fill the fragment with the value.
    Fill {
        frag: Variable<D>,
        value: Variable<D>,
    },
    /// Load the value into the fragment given the stride.
    Load {
        frag: Variable<D>,
        value: Variable<D>,
        offset: Variable<D>,
        stride: Variable<D>,
        layout: Option<FragmentLayout<D>>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        frag_a: Variable<D>,
        frag_b: Variable<D>,
        frag_c: Variable<D>,
        frag_d: Variable<D>,
        warp_size: u32,
    },
    /// Store the fragment in an output variable following the stride and the layout.
    Store {
        output: Variable<D>,
        frag: Variable<D>,
        stride: Variable<D>,
        offset: Variable<D>,
        layout: FragmentLayout<D>,
    },
    /// Cast
    Cast {
        input: Variable<D>,
        output: Variable<D>,
    },
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

impl<D: Dialect> Display for Fragment<D> {
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
    use super::*;

    pub fn compile_fragment_declaration<D: Dialect>(
        f: &mut std::fmt::Formatter<'_>,
        var: &Variable<D>,
    ) -> std::fmt::Result {
        match var {
            Variable::WmmaFragment { frag, .. } => writeln!(f, "{frag} {var};"),
            _ => panic!("variable must be a fragment"),
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
        fragment: &Fragment<D>,
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
                writeln!(f, "{namespace}::fill_fragment({frag}, {value});")
            }

            WmmaInstruction::Load {
                frag,
                value,
                stride,
                offset,
                layout: None,
            } => {
                let item = value.item();
                if item.vectorization > 1 {
                    let elem = item.elem;
                    let qualifier = value.const_qualifier();
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, reinterpret_cast<{elem}{qualifier}*>({value} + {offset}), {stride});"
                    )
                } else {
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, {value} + {offset}, {stride});"
                    )
                }
            }

            WmmaInstruction::Load {
                frag,
                value,
                offset,
                stride,
                layout: Some(layout),
            } => {
                let layout = match layout {
                    FragmentLayout::ColMajor => format!("{namespace}::mem_col_major"),
                    FragmentLayout::RowMajor => format!("{namespace}::mem_row_major"),
                    FragmentLayout::_Dialect(_) => "".to_string(),
                };
                let item = value.item();
                if item.vectorization > 1 {
                    let elem = item.elem;
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, reinterpret_cast<{elem} *>({value} + {offset}), {stride}, {layout});"
                    )
                } else {
                    writeln!(
                        f,
                        "{namespace}::load_matrix_sync({frag}, {value} + {offset}, {stride}, {layout});"
                    )
                }
            }

            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                ..
            } => writeln!(
                f,
                "{namespace}::mma_sync({frag_d}, {frag_a}, {frag_b}, {frag_c});"
            ),

            WmmaInstruction::Store {
                output,
                frag,
                stride,
                offset,
                layout,
            } => {
                let layout = match layout {
                    FragmentLayout::ColMajor => format!("{namespace}::mem_col_major"),
                    FragmentLayout::RowMajor => format!("{namespace}::mem_row_major"),
                    FragmentLayout::_Dialect(_) => "".to_string(),
                };

                let item = output.item();
                let mut reinterpret_cast = item.vectorization > 1;
                let elem = match item.elem {
                    Elem::BF16 => {
                        reinterpret_cast = true;
                        Elem::F16
                    }
                    _ => item.elem,
                };
                if reinterpret_cast {
                    writeln!(
                        f,
                        "{namespace}::store_matrix_sync(reinterpret_cast<{elem} *>({output} + {offset}), {frag}, {stride}, {layout});"
                    )
                } else {
                    writeln!(
                        f,
                        "{namespace}::store_matrix_sync({output} + {offset}, {frag}, {stride}, {layout});"
                    )
                }
            }
            WmmaInstruction::Cast { input, output } => {
                let ty = match output {
                    Variable::WmmaFragment { frag, .. } => frag.elem,
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
        }
    }
}

pub fn frag_as_ptr<D: Dialect>(
    f: &mut Formatter<'_>,
    frag: &Variable<D>,
    offset: &Variable<D>,
) -> Variable<D> {
    let item = frag.item();
    let mut frag_ptr = Variable::tmp_ptr(item);
    if frag.is_const() {
        frag_ptr.to_const();
    }
    let frag_ptr_out = frag_ptr.fmt_left();
    writeln!(f, "{frag_ptr_out} = {frag} + {offset};").unwrap();

    if item.vectorization > 1 {
        let mut item_value = item;
        item_value.vectorization = 1;
        frag_ptr.reinterpret_ptr(f, item_value)
    } else {
        frag_ptr
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

pub fn variable_to_frag<D: Dialect>(frag: &Variable<D>) -> Fragment<D> {
    match frag {
        Variable::WmmaFragment { frag, .. } => *frag,
        _ => panic!(),
    }
}
