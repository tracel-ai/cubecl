use core::{cell::Ref, fmt::Display};

use cubecl_core::ir::{
    Scope, dialect::InlineAsmOp, interfaces::TypedExt, prelude::*, types::VectorType,
};
use itertools::Itertools;
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{StringAttr, UnitAttr, VecAttr},
    opts::dce::SideEffects,
    printable::Printable,
};

use crate::{
    cuda::cuda_op,
    shared::{CppValue, lowering::LowerOp, scoped_block, ty::TypeExtCPP},
    target::Cuda,
};

/// Inline PTX. Restricted to zero or one results because C++ semantics are too hard otherwise.
/// Note that this does not *directly* map to PTX, because it actually destructures vectors to PTX
/// vector expressions automatically. This means more than one register can be returned if it's part
/// of a vector expression. To denote the difference, the syntax uses `$0`, `$1` etc for Pliron
/// values, as opposed to the usual `%0`, `%1` etc for the PTX registers.
#[pliron_op(
    name = "cuda.inline_ptx",
    format = "opt_attr($cuda_inline_ptx_volatile, $UnitAttr, label($volatile))
    attr($cuda_inline_ptx_ptx, $StringAttr) ` : ` types(CharSpace(`,`)) ` : ` operands(CharSpace(`,`))
    opt_attr($cuda_inline_ptx_clobbers, $VecAttr)",
    attributes = (cuda_inline_ptx_ptx: StringAttr, cuda_inline_ptx_volatile: UnitAttr, cuda_inline_ptx_clobbers: VecAttr),
    verifier = "succ"
)]
pub struct InlinePtxOp;

impl InlinePtxOp {
    pub fn new(
        ctx: &mut Context,
        result_ty: Option<TypeHandle>,
        ptx: impl Display,
        inputs: Vec<Value>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_ty.into_iter().collect(),
            inputs,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_cuda_inline_ptx_ptx(ctx, ptx.to_string().into());
        op
    }

    pub fn new_volatile(
        ctx: &mut Context,
        result_ty: Option<TypeHandle>,
        ptx: impl Display,
        inputs: Vec<Value>,
    ) -> Self {
        let op = Self::new(ctx, result_ty, ptx, inputs);
        op.set_attr_cuda_inline_ptx_volatile(ctx, UnitAttr::new());
        op
    }

    pub fn set_clobbers(&self, ctx: &Context, clobbers: Vec<String>) {
        if !clobbers.is_empty() {
            let clobbers = clobbers
                .into_iter()
                .map(StringAttr::new)
                .map(|attr| -> AttrObj { Box::new(attr) });
            self.set_attr_cuda_inline_ptx_clobbers(ctx, VecAttr(clobbers.collect()));
        }
    }

    pub fn raw_ptx<'a>(&self, ctx: &'a Context) -> Ref<'a, str> {
        Ref::map(self.get_attr_cuda_inline_ptx_ptx(ctx).unwrap(), |it| {
            it.as_str()
        })
    }

    pub fn is_volatile(&self, ctx: &Context) -> bool {
        self.get_attr_cuda_inline_ptx_volatile(ctx).is_some()
    }

    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }

    pub fn result(&self, ctx: &Context) -> Option<Value> {
        self.get_operation().deref(ctx).results().next()
    }
}

#[op_interface_impl]
impl SideEffects for InlinePtxOp {
    fn has_side_effects(&self, ctx: &Context) -> bool {
        self.get_attr_cuda_inline_ptx_volatile(ctx).is_some()
    }
}

#[macro_export]
macro_rules! ptx_block {
    ($($lines: expr)*) => {{
        let mut out = String::from("{\n\t");
        $(
            out.push_str(&$lines);
            out.push_str("\n\t");
        )*
        out.push_str("}");
        out
    }};
}

cuda_op!(InlinePtxOp, |op, ctx| {
    let mut ptx = op.raw_ptx(ctx).to_owned();
    let result = op.result(ctx);
    let inputs = op.inputs(ctx);

    let mut ptx_idx = 0;
    let mut plir_idx = 0;

    if let Some(result) = result {
        ptx = insert_placeholders(ctx, &ptx, result.get_type(ctx), plir_idx, &mut ptx_idx);
        plir_idx += 1;
    }

    for input in inputs.iter() {
        ptx = insert_placeholders(ctx, &ptx, input.get_type(ctx), plir_idx, &mut ptx_idx);
        plir_idx += 1;
    }

    let out_regs = result
        .iter()
        .flat_map(|val| flatten_result(ctx, val.get_type(ctx)))
        .join(", ");
    let input_regs = inputs
        .iter()
        .flat_map(|val| flatten_operand(ctx, *val))
        .join(", ");

    let volatile = if op.is_volatile(ctx) { "volatile" } else { "" };
    let clobbers = if let Some(clobbers) = op.get_attr_cuda_inline_ptx_clobbers(ctx) {
        let names = clobbers.0.iter();
        let names = names
            .map(|it| it.downcast_ref::<StringAttr>().unwrap().as_str())
            .map(|name| format!(r#""{name}""#))
            .join(", ");
        format!(": {names}")
    } else {
        String::new()
    };

    let asm = format!("asm {volatile}({ptx:?} : {out_regs} : {input_regs} {clobbers});",);

    if let Some(result) = result {
        let block = scoped_block!(
            format!("{} result;", result.get_type(ctx).to_cpp(ctx))
            asm
            format!("return result;")
        );
        format!("{} = {block};", result.fmt_left(ctx))
    } else {
        asm
    }
});

fn flatten_result(ctx: &Context, ty: TypeHandle) -> Vec<String> {
    if ty.is_vector(ctx) {
        let vec = ty.vector_size(ctx);
        let constraint = infer_constraint_letter(ctx, ty.scalar_ty(ctx));
        (0..vec)
            .map(|i| format!(r#""={constraint}"(result.i_{i})"#))
            .collect()
    } else {
        let constraint = infer_constraint_letter(ctx, ty.get_type(ctx));
        vec![format!(r#""={constraint}"(result)"#)]
    }
}

fn flatten_operand(ctx: &Context, val: Value) -> Vec<String> {
    if val.get_type(ctx).deref(ctx).is::<VectorType>() {
        let vec = val.vector_size(ctx);
        let constraint = infer_constraint_letter(ctx, val.scalar_ty(ctx));
        (0..vec)
            .map(|i| format!(r#""{constraint}"({}.i_{i})"#, val.name(ctx)))
            .collect()
    } else {
        let constraint = infer_constraint_letter(ctx, val.get_type(ctx));
        vec![format!(r#""{constraint}"({})"#, val.name(ctx))]
    }
}

fn insert_placeholders(
    ctx: &Context,
    ptx: &str,
    ty: TypeHandle,
    plir_idx: usize,
    ptx_idx: &mut usize,
) -> String {
    let pat = format!("${plir_idx}");
    if !ptx.contains(&pat) {
        panic!("Tried substituting argument {pat} in PTX, but it wasn't found.")
    }
    let substitute = if ty.deref(ctx).is::<VectorType>() {
        let vec = ty.vector_size(ctx);
        let mut placeholders = (0..vec).map(|i| format!("%{}", *ptx_idx + i));
        let substitute = format!("{{{}}}", placeholders.join(", "));
        *ptx_idx += vec;
        substitute
    } else {
        let placeholder = format!("%{ptx_idx}");
        *ptx_idx += 1;
        placeholder
    };
    ptx.replace(&pat, &substitute)
}

fn infer_constraint_letter(ctx: &Context, ty: TypeHandle) -> char {
    if ty.is_bool(ctx) {
        'b'
    } else if ty.is_int_of_width(ctx, 16) || ty.is_uint_of_width(ctx, 16) {
        'h'
    } else if ty.is_int_of_width(ctx, 32) || ty.is_uint_of_width(ctx, 32) {
        'r'
    } else if ty.is_int_of_width(ctx, 64) || ty.is_uint_of_width(ctx, 64) {
        'l'
    } else if ty.is_float32(ctx) {
        'f'
    } else if ty.is_float64(ctx) {
        'd'
    } else if ty.is_ptr(ctx) {
        'l'
    } else {
        panic!(
         "The register type could not be deduced from Pliron type. The type {} is not supported. 
Supported types are: bool, i16, i32, i64, f32, f64, pointers.
Please use cube.reinterpret_cast if you have different type.
See the constraints from here: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints",
        ty.disp(ctx));
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for InlineAsmOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let ptx = self.asm(ctx).as_str().to_owned();
        let inputs = self.inputs(ctx);
        let results = self
            .get_operation()
            .opt_result(ctx)
            .map(|res| res.get_type(ctx));
        let inline_ptx = InlinePtxOp::new(ctx, results, ptx, inputs);
        if !self.pure(ctx) {
            inline_ptx.set_attr_cuda_inline_ptx_volatile(ctx, UnitAttr::new());
        }
        if !self.nomem(ctx) {
            inline_ptx.set_attr_cuda_inline_ptx_clobbers(
                ctx,
                VecAttr(vec![Box::new(StringAttr::new("memory".into()))]),
            );
        }
        inline_ptx
            .get_operation()
            .insert_before(ctx, self.get_operation());
        inline_ptx.get_operation().results(ctx)
    }
}
