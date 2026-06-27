use std::fmt::Formatter;

use crate::{
    hip::arch::AMDArchitecture,
    shared::{
        Architecture, CompilationOptions, CppValue, SupportedMmaCombinations, frag_ident_str,
        frag_layout_str,
        ty::{TypeExtCPP, TypedExtCPP},
    },
};
use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        ContextExt, ElemType, FloatKind,
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        features::MmaConfig,
        interfaces::TypedExt,
        types::MatrixScope,
    },
};
use pliron::{
    context::Context,
    printable::Printable,
    r#type::{Type, TypeHandle, Typed},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaIntrinsicCompiler {}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaFill {
    frag: MatrixType,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaLoad {
    frag: MatrixType,
    layout: Option<MatrixLayout>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaStore {
    frag: MatrixType,
    layout: MatrixLayout,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaExecute {
    pub frag_a: MatrixType,
    pub frag_b: MatrixType,
    pub frag_c: MatrixType,
    pub frag_d: MatrixType,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaCast {
    frag_input: MatrixType,
    frag_output: MatrixType,
}

impl WmmaFill {
    pub fn fn_name(&self, ctx: &Context) -> String {
        let layout = frag_layout_str(Some(&self.frag.layout));
        let ident = frag_ident_str(&self.frag.ident);
        let MatrixShape { m, n, k } = self.frag.shape;
        let elem = self.frag.elem_ty.to_cpp(ctx);

        format!("wmma_fill_{elem}_{ident}_{m}x{n}x{k}_{layout}",)
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>, ctx: &Context) -> std::fmt::Result {
        let elem = self.frag.elem_ty.to_cpp(ctx);
        let frag = self.frag.get_self_handle(ctx).to_cpp(ctx);
        let name = self.fn_name(ctx);

        write!(
            f,
            "
// Fill the fragment.
__device__ void {name}({frag}& frag, {elem} value) {{
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {{
      frag[i] = value;
    }}
}}
        "
        )
    }
}

impl WmmaLoad {
    pub fn fn_name(&self, ctx: &Context) -> String {
        let layout_frag = frag_layout_str(Some(&self.frag.layout));
        let layout = frag_layout_str(self.layout.as_ref());
        let ident = frag_ident_str(&self.frag.ident);
        let elem = self.frag.elem_ty.to_cpp(ctx);
        let MatrixShape { m, n, k } = self.frag.shape;

        format!("wmma_load_{elem}_{ident}_{m}x{n}x{k}_{layout_frag}_{layout}",)
    }

    /// Matrix A must be in column major layout (so fragments correspond to a row)
    /// Matrices B, C and D must be in row major layout (so fragments correspond to a column)
    ///
    /// Each lane is a thread so each column get 8 VGPRs used to store fragments
    /// Here is the layout for C and D matrices and how they map to registers
    ///
    /// Lane index   0      1      2      3      ...     13     14     15     ...     17     18     ...     30     31
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR0      | 1,1  | 1,2  | 1,3  | 1,4  | ...  | 1,13 | 1,14 | 1,15 | ...  | 2,1  | 2,2  | ...  | 2,15 | 2,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR1      | 3,1  | 3,2  | 3,3  | 3,4  | ...  | 3,13 | 3,14 | 3,15 | ...  | 4,1  | 4,2  | ...  | 4,15 | 4,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR2      | 5,1  | 5,2  | 5,3  | 5,4  | ...  | 5,13 | 5,14 | 5,15 | ...  | 6,1  | 6,2  | ...  | 6,15 | 6,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR3      | 7,1  | 7,2  | 7,3  | 7,4  | ...  | 7,13 | 7,14 | 7,15 | ...  | 8,1  | 8,2  | ...  | 8,15 | 8,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR4      | 9,1  | 9,2  | 9,3  | 9,4  | ...  | 9,13 | 9,14 | 9,15 | ...  | 10,1 | 10,2 | ...  | 10,15| 10,16|
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR5      | 11,1 | 11,2 | 11,3 | 11,4 | ...  | 11,13| 11,14| 11,15| ...  | 12,1 | 12,2 | ...  | 12,15| 12,16|
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR6      | 13,1 | 13,2 | 13,3 | 13,4 | ...  | 13,13| 13,14| 13,15| ...  | 14,1 | 14,2 | ...  | 14,15| 14,16|
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR7      | 15,1 | 15,2 | 15,3 | 15,4 | ...  | 15,13| 15,14| 15,15| ...  | 16,1 | 16,2 | ...  | 16,15| 16,16|
    /// --------------------------------------------------------------------------------------------------------------
    pub fn format_extension(&self, f: &mut Formatter<'_>, ctx: &Context) -> std::fmt::Result {
        let elem = self.frag.elem_ty;
        let frag = self.frag;
        let name = self.fn_name(ctx);

        let (index_body, length, step) = match frag.ident {
            MatrixIdent::A | MatrixIdent::B => {
                let length = 16;
                let step = 1;
                // fragment a and b are always in half precision and they don't require special attention
                // to how they are stored in memory as matrix A and B are also in half precision
                let index = if (frag.ident == MatrixIdent::A
                    && frag.layout == MatrixLayout::ColMajor)
                    || (frag.ident == MatrixIdent::B && frag.layout == MatrixLayout::RowMajor)
                {
                    "i * stride + wmmaLane".to_string()
                } else {
                    "i + wmmaLane * stride".to_string()
                };
                (index, length, step)
            }
            MatrixIdent::Accumulator => {
                let length = 8;
                let step = get_output_accumulator_index_step(ctx, elem, &frag);
                let index = match self.layout {
                    Some(MatrixLayout::ColMajor) => {
                        "(i * uint(2) + threadIdx.x / uint(16)) + wmmaLane * stride".to_string()
                    }
                    Some(MatrixLayout::RowMajor) => {
                        "(i * uint(2) + threadIdx.x / uint(16)) * stride + wmmaLane".to_string()
                    }
                    _ => panic!(
                        "cannot load data to an accumulator without knowing the layout of the data"
                    ),
                };
                (index, length, step)
            }
        };
        let frag = frag.get_self_handle(ctx).to_cpp(ctx);
        let elem = elem.to_cpp(ctx);

        write!(
            f,
            "
// Load the fragment.
__device__ void {name}({frag}& frag, const {elem}* value_ptr, const uint stride) {{
    {WMMA_LANE_DEF}

    #pragma unroll
    for (uint i = 0; i < {length}; ++i) {{
      const uint index = {index_body};
      frag[i * {step}] = value_ptr[index];
    }}
}}
        "
        )
    }
}

impl WmmaStore {
    pub fn fn_name(&self, ctx: &Context) -> String {
        let layout_frag = frag_layout_str(Some(&self.frag.layout));
        let layout = frag_layout_str(Some(&self.layout));
        let ident = frag_ident_str(&self.frag.ident);
        let MatrixShape { m, n, k } = self.frag.shape;
        let elem = self.frag.elem_ty.to_cpp(ctx);

        format!("wmma_store_{elem}_{ident}_{m}x{n}x{k}_{layout_frag}_{layout}",)
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>, ctx: &Context) -> std::fmt::Result {
        let elem = self.frag.elem_ty;
        let frag = self.frag;
        let name = self.fn_name(ctx);
        // frag holds a result column where threads 0-15 of the wavefront have the even rows and threads 16-31 the odd rows
        // moreover, since we use OPSEL to false in the Execute instruction in f16 output format, the output elements are
        // stored in even indexes (0, 2, 4, ...) (low 16-bits of the VGPR) in frag
        let frag_idx = if elem.is_half(ctx) {
            "elemIdx * 2"
        } else {
            "elemIdx"
        };
        // FragmentLayout here represents the desired layout of the matrix C
        let output_idx = match self.layout {
            MatrixLayout::ColMajor => "wmmaLane * stride + rowIdx".to_string(),
            MatrixLayout::RowMajor => "wmmaLane + rowIdx * stride".to_string(),
            _ => unreachable!(),
        };

        let frag = frag.get_self_handle(ctx).to_cpp(ctx);
        let elem = elem.to_cpp(ctx);

        write!(
            f,
            "
// Store the fragment.
__device__ void {name}(const {frag}& frag, {elem}* output_ptr, uint stride) {{
    {WMMA_LANE_DEF}

    #pragma unroll
    for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {{
      const uint rowIdx = elemIdx * uint(2) + threadIdx.x / uint(16);
      output_ptr[{output_idx}] = frag[{frag_idx}];
    }}
}}
        "
        )
    }
}

impl WmmaExecute {
    pub fn from_manual(shape: MatrixShape, ab_elem: TypeHandle, cd_elem: TypeHandle) -> Self {
        // Hack, remove once types no longer need a mutable context
        let frag_a = MatrixType::new(
            MatrixIdent::A,
            shape,
            ab_elem,
            MatrixLayout::ColMajor,
            MatrixScope::Plane,
        );
        let frag_b = MatrixType::new(
            MatrixIdent::B,
            shape,
            ab_elem,
            MatrixLayout::RowMajor,
            MatrixScope::Plane,
        );
        let frag_cd = MatrixType::new(
            MatrixIdent::Accumulator,
            shape,
            cd_elem,
            MatrixLayout::RowMajor,
            MatrixScope::Plane,
        );

        WmmaExecute::new(frag_a, frag_b, frag_cd, frag_cd)
    }

    pub fn fn_name(&self, ctx: &Context) -> String {
        format!(
            "wmma_execute_16x16x16_{}_{}",
            self.frag_a.elem_ty.to_cpp(ctx),
            self.frag_c.elem_ty.to_cpp(ctx)
        )
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>, ctx: &Context) -> std::fmt::Result {
        let name = self.fn_name(ctx);
        let ab_format = if self.frag_a.elem_ty.is_float32(ctx) {
            "f32"
        } else if self.frag_a.elem_ty.is_bfloat16(ctx) {
            "bf16"
        } else {
            "f16"
        };
        let (cd_format, opsel) = if self.frag_c.elem_ty.is_float32(ctx) {
            ("f32", "")
        } else if self.frag_a.elem_ty.is_bfloat16(ctx) {
            ("bf16", ", false")
        } else {
            ("f16", ", false")
        };
        let warp_size = 32;
        write!(
            f,
            "
// Execute wmma.
__device__ void {name}(const {}& frag_a, const {}& frag_b, const {}& frag_c, {}& frag_d) {{
    frag_d = __builtin_amdgcn_wmma_{cd_format}_16x16x16_{ab_format}_w{warp_size}(frag_a, frag_b, frag_c{opsel});
}}
        ",
            self.frag_a.get_self_handle(ctx).to_cpp(ctx),
            self.frag_b.get_self_handle(ctx).to_cpp(ctx),
            self.frag_c.get_self_handle(ctx).to_cpp(ctx),
            self.frag_d.get_self_handle(ctx).to_cpp(ctx)
        )
    }
}

impl WmmaCast {
    pub fn fn_name(&self, ctx: &Context) -> String {
        let layout = frag_layout_str(Some(&self.frag_input.layout));
        let ident = frag_ident_str(&self.frag_input.ident);
        let MatrixShape { m, n, k } = self.frag_input.shape;
        let elem = self.frag_input.elem_ty.to_cpp(ctx);
        let elem_out = self.frag_output.elem_ty.to_cpp(ctx);

        format!("wmma_cast_{elem}_to_{elem_out}_{ident}_{m}x{n}x{k}_{layout}",)
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>, ctx: &Context) -> std::fmt::Result {
        let input = self.frag_input.get_self_handle(ctx).to_cpp(ctx);
        let output = self.frag_output.get_self_handle(ctx).to_cpp(ctx);
        let name = self.fn_name(ctx);
        let step = match self.frag_output.ident {
            MatrixIdent::Accumulator => {
                get_output_accumulator_index_step(ctx, self.frag_input.elem_ty, &self.frag_output)
            }
            _ => 1,
        };

        write!(
            f,
            "
// Cast the fragment.
__device__ void {name}(const {input}& input, {output}& output) {{
    #pragma unroll
    for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {{
      output[elemIdx * {step}] = input[elemIdx];
    }}
}}
        "
        )
    }
}

pub(super) fn compile_fragment_intrinsic(ctx: &Context, mat_ty: &MatrixType) -> String {
    match mat_ty.ident {
        MatrixIdent::A | MatrixIdent::B => {
            if mat_ty.elem_ty.is_float16(ctx) {
                "half16_t".into()
            } else if mat_ty.elem_ty.is_bfloat16(ctx) {
                "bhalf16_t".into()
            } else {
                panic!(
                    "unsupported type {} for {}",
                    mat_ty.elem_ty.disp(ctx),
                    mat_ty.disp(ctx)
                )
            }
        }
        MatrixIdent::Accumulator => {
            if mat_ty.elem_ty.is_float16(ctx) {
                "half16_t".into()
            } else if mat_ty.elem_ty.is_bfloat16(ctx) {
                "bhalf16_t".into()
            } else if mat_ty.elem_ty.is_float32(ctx) {
                "float8_t".into()
            } else {
                panic!(
                    "unsupported type {} for {}",
                    mat_ty.elem_ty.disp(ctx),
                    mat_ty.disp(ctx)
                )
            }
        }
    }
}

pub(super) fn compile_fill_intrinsic(ctx: &Context, op: &FillOp) -> String {
    let matrix = op.matrix(ctx);
    let value = op.value(ctx).name(ctx);
    let extension = WmmaFill::new(*matrix.get_type(ctx).deref(ctx).downcast_ref().unwrap());
    let name = extension.fn_name(ctx);
    format!("{name}(*{}, {value});", matrix.name(ctx))
}

pub(super) fn compile_load_intrinsic(ctx: &Context, op: &LoadOp) -> String {
    let mat = op.matrix(ctx);
    let value_ptr = op.source(ctx).name(ctx);
    let stride = op.stride(ctx).name(ctx);
    let mat_ty = *mat.get_type(ctx).deref(ctx).downcast_ref().unwrap();
    let layout = op.layout(ctx).map(|it| it.0);
    let extension = WmmaLoad::new(mat_ty, layout);
    let name = extension.fn_name(ctx);
    format!("{name}(*{}, {value_ptr}, {stride});", mat.name(ctx))
}

pub(super) fn compile_store_intrinsic(ctx: &Context, op: &StoreOp) -> String {
    let mat = op.matrix(ctx);
    let output_ptr = op.destination(ctx).name(ctx);
    let stride = op.stride(ctx).name(ctx);
    let mat_ty = *mat.get_type(ctx).deref(ctx).downcast_ref().unwrap();
    let layout = op.layout(ctx).0;
    let extension = WmmaStore::new(mat_ty, layout);
    let name = extension.fn_name(ctx);
    format!("{name}(*{}, {output_ptr}, {stride});", mat.name(ctx))
}

pub(super) fn compile_execute_intrinsic(ctx: &Context, op: &MultiplyAccumulateOp) -> String {
    let warp_size = ctx.aux_ty::<CompilationOptions>().warp_size;
    if warp_size != 32 {
        panic!("Only warp size of 32 supported for Wmma::Execute on HIP");
    }

    let frag_a = op.mat_a(ctx);
    let frag_b = op.mat_b(ctx);
    let frag_c = op.mat_c(ctx);
    let frag_d = op.mat_d(ctx);

    let extension = WmmaExecute::new(
        *frag_a.get_type(ctx).deref(ctx).downcast_ref().unwrap(),
        *frag_b.get_type(ctx).deref(ctx).downcast_ref().unwrap(),
        *frag_c.get_type(ctx).deref(ctx).downcast_ref().unwrap(),
        *frag_d.get_type(ctx).deref(ctx).downcast_ref().unwrap(),
    );
    let name = extension.fn_name(ctx);
    format!(
        "{name}({}, {}, {}, {});",
        frag_a.name(ctx),
        frag_b.name(ctx),
        frag_c.name(ctx),
        frag_d.name(ctx)
    )
}

pub(super) fn compile_cast_intrinsic(ctx: &Context, op: &CastOp) -> String {
    let input = op.input(ctx);
    let output = op.output(ctx);

    let extension = WmmaCast::new(
        *input.get_type(ctx).deref(ctx).downcast_ref().unwrap(),
        *output.get_type(ctx).deref(ctx).downcast_ref().unwrap(),
    );
    let name = extension.fn_name(ctx);
    let input = input.name(ctx);
    let output = output.name(ctx);
    format!("{name}({input}, {output});")
}

pub(super) fn supported_wmma_combinations_intrinsic(
    arch: &AMDArchitecture,
) -> SupportedMmaCombinations {
    // Reference: https://gpuopen.com/learn/wmma_on_rdna3/
    let mut result: SupportedMmaCombinations = vec![];
    if arch.is_wmma_capable() {
        // Types fully supported.
        let types = vec![
            (
                ElemType::Float(FloatKind::F16), // m
                ElemType::Float(FloatKind::F16), // n
                ElemType::Float(FloatKind::F16), // k
            ),
            (
                ElemType::Float(FloatKind::F16),
                ElemType::Float(FloatKind::F16),
                ElemType::Float(FloatKind::F32),
            ),
            (
                ElemType::Float(FloatKind::BF16),
                ElemType::Float(FloatKind::BF16),
                ElemType::Float(FloatKind::F32),
            ),
        ];
        let combinations: SupportedMmaCombinations = types
            .into_iter()
            .map(|(a, b, c)| MmaConfig {
                a_type: a,
                b_type: b,
                cd_type: c,
                m: 16,
                n: 16,
                k: 16,
            })
            .collect();
        result.extend(combinations);
    }
    result
}

fn get_output_accumulator_index_step(
    ctx: &Context,
    input_elem: TypeHandle,
    output: &MatrixType,
) -> u32 {
    // Each VGPR is 32 bit wide and there is 8 VGPR per lane, an accumulator can then be either:
    // - a vector of 8 float
    // - a vector of 16 half
    // Depending on the precision used for the input, the whole 32 bits per register will be used or
    // just only 16 bits. In such a case we always use the lower 16 bits (opsel set to false) which means
    // that we only assign values to even indexes of the accumulator (0, 2, 4, ...)

    assert_eq!(output.ident, MatrixIdent::Accumulator);
    assert!(input_elem.is_half(ctx) || input_elem.is_float32(ctx));

    if output.elem_ty.is_half(ctx) { 2 } else { 1 }
}

// threads 0-15 and threads 16-31 of the wavefront hold the same fragments respectively
// in other words fragments are duplicated
// so lanes 0,16 / 1,17 / ... / 15, 31 are the same
static WMMA_LANE_DEF: &str = "uint wmmaLane = uint(threadIdx.x % 16);";
