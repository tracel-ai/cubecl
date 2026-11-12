//! Cooperative matrices other than the 8x8 simdgroup matrix, on Metal 4's tensor operations:
//! the `matmul2d` of Metal Performance Primitives, which Apple10 GPUs run on their neural
//! accelerators.
//!
//! Each fragment is a `cooperative_tensor` of a `matmul2d` that one SIMD group runs: its left
//! input, right input or destination. The operation is named by the fragment's shape, the element
//! types of the product's operands, and whether each input is column major, which the operation
//! reads transposed (a tensor view's innermost stride is 1). It spreads the cells over the lanes
//! in a way it does not specify, so fragments are loaded and stored through tensor views of
//! memory, and combined cell by cell only with fragments of the same shape and use.
//!
//! A fragment's type names the element types of the products it takes part in, which therefore
//! have to agree, and its own layout. Each product runs the operation its own operands name.

use core::fmt::Display;

use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        AddressSpace, ElemType, FloatKind, IntKind, UIntKind,
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        features::MmaConfig,
        interfaces::{ScalarType, TypeExt},
        prelude::*,
    },
};

use crate::{
    error::{CompileError, EmissionErrors},
    metal::{supported_tensor_op_combinations_metal, ty::ptr_space},
    shared::{CppValue, DeclareMatrixOp, ty::TypeExtCPP, wmma_api_base::matrix_ty},
};

/// Whether the fragment shape is a tensor operation's rather than the simdgroup matrix's.
pub fn is_tensor_op(shape: &MatrixShape) -> bool {
    (shape.m, shape.n, shape.k) != (8, 8, 8)
}

/// Fails the compilation with `reason`; what it emits is never compiled.
fn unsupported(ctx: &Context, reason: String) -> String {
    ctx.aux_ty::<EmissionErrors>()
        .record(CompileError::Unsupported(reason));
    String::new()
}

/// The element type of a fragment's cells.
fn elem(ctx: &Context, ty: &MatrixType) -> ElemType {
    let elem = ty.elem_ty.deref(ctx);
    type_cast::<dyn ScalarType>(&*elem)
        .expect("a fragment holds scalars")
        .elem_type(ctx)
}

/// The MSL name of an element a tensor operation takes.
fn msl(elem: ElemType) -> &'static str {
    match elem {
        ElemType::Float(FloatKind::F16) => "half",
        ElemType::Float(FloatKind::BF16) => "bfloat",
        ElemType::Float(FloatKind::F32) => "float",
        ElemType::Int(IntKind::I8) => "int8_t",
        ElemType::Int(IntKind::I32) => "int32_t",
        ElemType::UInt(UIntKind::U8) => "uint8_t",
        _ => unreachable!("only supported products are emitted"),
    }
}

/// The element types of a product's `A`, `B` and accumulator.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Product([ElemType; 3]);

impl Display for Product {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [a, b, cd] = self.0;
        write!(f, "{a} x {b} -> {cd}")
    }
}

impl Product {
    /// The product `mma` computes.
    fn of_mma(ctx: &Context, mma: &MultiplyAccumulateOp) -> Self {
        let [a, b, c] = [mma.mat_a(ctx), mma.mat_b(ctx), mma.mat_c(ctx)];
        Self([a, b, c].map(|operand| elem(ctx, &matrix_ty(ctx, operand))))
    }

    /// The product of a fragment that takes part in none: an 8-bit integer input's into `i32`,
    /// a float input's into `f32`, an `i32` accumulator's from `i8`, an `f32` one's from `f16`, a
    /// 16-bit float one's from its own type.
    fn assumed(ident: MatrixIdent, elem: ElemType) -> Self {
        let (f16, f32, i32) = (
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F32),
            ElemType::Int(IntKind::I32),
        );
        let [ab, cd] = match (ident, elem) {
            (MatrixIdent::Accumulator, ElemType::Int(_)) => [ElemType::Int(IntKind::I8), elem],
            (MatrixIdent::Accumulator, ElemType::Float(FloatKind::F32)) => [f16, elem],
            (MatrixIdent::Accumulator, _) => [elem, elem],
            (_, ElemType::Int(_) | ElemType::UInt(_)) => [elem, i32],
            (_, _) => [elem, f32],
        };
        Self([ab, ab, cd])
    }

    /// The product every product `matrix` takes part in computes, or the one it is
    /// [`assumed`](Self::assumed) to take part in. Products of different element types fail: a
    /// fragment is the operand of one operation.
    fn of(ctx: &Context, matrix: Value) -> core::result::Result<Self, String> {
        let ty = matrix_ty(ctx, matrix);
        let uses = matrix.uses(ctx);
        let mut products = uses
            .iter()
            .filter_map(|r#use| r#use.user_op().as_op::<MultiplyAccumulateOp>(ctx))
            .map(|mma| Self::of_mma(ctx, &mma));
        let Some(product) = products.next() else {
            return Ok(Self::assumed(ty.ident, elem(ctx, &ty)));
        };
        match products.find(|other| *other != product) {
            None => Ok(product),
            Some(other) => Err(format!(
                "one {:?} fragment in a {product} product and a {other} one: a Metal tensor \
                 operation's fragment belongs to products of one element type",
                ty.ident
            )),
        }
    }

    /// Fails unless the device's tensor operations run this product on `shape`.
    fn check(&self, shape: &MatrixShape) -> core::result::Result<(), String> {
        let [a_type, b_type, cd_type] = self.0;
        let config = MmaConfig {
            a_type,
            b_type,
            cd_type,
            m: shape.m as u32,
            n: shape.n as u32,
            k: shape.k as u32,
        };
        match supported_tensor_op_combinations_metal().contains(&config) {
            true => Ok(()),
            false => Err(format!(
                "Metal tensor operations run no {}x{}x{} {self} product",
                shape.m, shape.n, shape.k
            )),
        }
    }

    /// The `matmul2d` of one SIMD group on a `shape` tile, its inputs read `transposed`.
    fn operation(shape: &MatrixShape, [left, right]: [bool; 2]) -> String {
        let MatrixShape { m, n, k } = shape;
        format!(
            "mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor({m}, {n}, {k}, \
             {left}, {right}, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), \
             metal::execution_simdgroup>"
        )
    }

    /// The cooperative tensor of a `ty` fragment of this product. An input's names its own
    /// layout; the lanes of either input and of the destination hold the same cells whichever
    /// input is transposed, so the product a fragment takes part in can transpose the other.
    fn fragment(&self, ty: &MatrixType) -> String {
        let column = ty.layout == MatrixLayout::ColMajor;
        let transposed = match ty.ident {
            MatrixIdent::A => [column, false],
            MatrixIdent::B => [false, column],
            MatrixIdent::Accumulator => [false, false],
        };
        let op = Self::operation(&ty.shape, transposed);
        let [a, b, cd] = self.0.map(msl);
        let left = format!("{op}::cooperative_tensor_left_input_t<{a}, {b}, {cd}>");
        let right = format!("{op}::cooperative_tensor_right_input_t<{a}, {b}, {cd}>");
        match ty.ident {
            MatrixIdent::A => left,
            MatrixIdent::B => right,
            MatrixIdent::Accumulator => {
                format!("{op}::cooperative_tensor_destination_t<{left}, {right}, {cd}>")
            }
        }
    }
}

/// The cooperative tensor of a `ty` fragment, which only its declaration names: it depends on
/// the products the fragment takes part in.
pub fn fragment_ty(ctx: &Context, ty: &MatrixType) -> String {
    unsupported(
        ctx,
        format!(
            "the type of a {:?} tensor-op fragment outside its declaration",
            ty.ident
        ),
    )
}

pub fn declare(ctx: &Context, op: &DeclareMatrixOp) -> String {
    let matrix = op.get_result(ctx);
    let ty = matrix_ty(ctx, matrix);
    let product = Product::of(ctx, matrix).and_then(|product| {
        product.check(&ty.shape)?;
        Ok(product)
    });
    let ty = match product {
        Ok(product) => product.fragment(&ty),
        Err(reason) => return unsupported(ctx, reason),
    };
    let id = matrix.name(ctx);
    format!("{ty} {id}_store;\nthread {ty}* {id} = &{id}_store;\n")
}

pub fn fill(ctx: &Context, op: &FillOp) -> String {
    let elem = matrix_ty(ctx, op.matrix(ctx)).elem_ty.to_cpp(ctx);
    let value = format!("{elem}({})", op.value(ctx).name(ctx));
    cells(op.matrix(ctx).name(ctx), |cell| {
        format!("{cell} = {value};")
    })
}

/// `body` on every cell the lane holds of `matrix`, named by the expression it is given.
fn cells(matrix: impl Display, body: impl Fn(&str) -> String) -> String {
    let body = body(&format!("(*{matrix})[i]"));
    format!("for (uint16_t i = 0; i < {matrix}->get_capacity(); ++i) {{ {body} }}\n")
}

/// A fragment's cells in memory, at `ptr` with `stride` elements between the starts of its rows
/// (row major) or columns (column major).
enum Memory {
    /// A tensor view of them: an `A` or `B` fragment's (a column major one transposed), a row
    /// major accumulator's.
    View(String),
    /// The cell at the multidimensional index `at` of a column major accumulator, which no view
    /// lays out with an innermost stride of 1.
    Cell(String),
}

impl Memory {
    /// The cells of a `ty` fragment at `ptr`: an accumulator's laid out as `layout` says, an
    /// input's as it was declared, which names the operation that reads it (as on CUDA, and as the
    /// simdgroup matrix loads).
    fn new(
        ctx: &Context,
        ty: &MatrixType,
        ptr: Value,
        stride: Value,
        layout: MatrixLayout,
    ) -> Self {
        let layout = match ty.ident {
            MatrixIdent::Accumulator => layout,
            MatrixIdent::A | MatrixIdent::B => ty.layout,
        };
        let MatrixShape { m, n, k } = ty.shape;
        let (rows, columns) = match ty.ident {
            MatrixIdent::A => (m, k),
            MatrixIdent::B => (k, n),
            MatrixIdent::Accumulator => (m, n),
        };
        let space = ptr_space(ptr.get_type(ctx).as_ptr(ctx).address_space);
        let elem = ty.elem_ty.to_cpp(ctx);
        let stride = stride.name(ctx);
        let ptr = format!("(({space} {elem}*)({}))", ptr.name(ctx));
        // innermost extent first
        let extents = match (ty.ident, layout) {
            (MatrixIdent::Accumulator, MatrixLayout::ColMajor) => {
                return Self::Cell(format!("{ptr}[at[0] * int({stride}) + at[1]]"));
            }
            (_, MatrixLayout::ColMajor) => format!("metal::extents<int32_t, {rows}, {columns}>"),
            _ => format!("metal::extents<int32_t, {columns}, {rows}>"),
        };
        Self::View(format!(
            "metal::tensor<{space} {elem}, {extents}, metal::tensor_inline>({ptr}, {extents}(), \
             metal::array<int32_t, 2>{{1, int({stride})}})"
        ))
    }

    /// `view` with the view, else `cell` on every valid cell and its place in memory.
    fn access(
        self,
        matrix: impl Display,
        view: impl Fn(&str) -> String,
        cell: impl Fn(&str, &str) -> String,
    ) -> String {
        match self {
            Self::View(memory) => view(&memory),
            Self::Cell(memory) => cells(&matrix, |value| {
                format!(
                    "if ({matrix}->is_valid_element(i)) {{ auto at = \
                     {matrix}->get_multidimensional_index(i); {} }}",
                    cell(value, &memory)
                )
            }),
        }
    }
}

pub fn load(ctx: &Context, op: &LoadOp) -> String {
    let matrix = op.matrix(ctx);
    let ty = matrix_ty(ctx, matrix);
    let memory = Memory::new(ctx, &ty, op.source(ctx), op.stride(ctx), op.layout(ctx).0);
    let name = matrix.name(ctx);
    memory.access(
        &name,
        |view| format!("{name}->load({view});\n"),
        |value, cell| format!("{value} = {cell};"),
    )
}

/// A store to threadgroup memory ends in a SIMD-group barrier, as the simdgroup matrix's does,
/// so every lane reads the cells the others stored.
pub fn store(ctx: &Context, op: &StoreOp) -> String {
    let matrix = op.matrix(ctx);
    let ty = matrix_ty(ctx, matrix);
    let destination = op.destination(ctx);
    let memory = Memory::new(ctx, &ty, destination, op.stride(ctx), op.layout(ctx).0);
    let name = matrix.name(ctx);
    let store = memory.access(
        &name,
        |view| format!("{name}->store({view});\n"),
        |value, cell| format!("{cell} = {value};"),
    );
    match destination.get_type(ctx).as_ptr(ctx).address_space {
        AddressSpace::Shared => format!("{store}simdgroup_barrier(mem_flags::mem_threadgroup);\n"),
        _ => store,
    }
}

/// `d = a b + c`: the operation accumulates into its destination, so `c` is first copied to a
/// distinct `d`, cell by cell, since cooperative tensors are not assignable.
pub fn execute(ctx: &Context, op: &MultiplyAccumulateOp) -> String {
    let [a, b, c, d] = [op.mat_a(ctx), op.mat_b(ctx), op.mat_c(ctx), op.mat_d(ctx)];
    let shape = matrix_ty(ctx, d).shape;
    if let Err(reason) = Product::of_mma(ctx, op).check(&shape) {
        return unsupported(ctx, reason);
    }
    let column = |matrix| matrix_ty(ctx, matrix).layout == MatrixLayout::ColMajor;
    let operation = Product::operation(&shape, [column(a), column(b)]);
    let copy = match c == d {
        true => String::new(),
        false => copy(c.name(ctx), d.name(ctx), ""),
    };
    let [a, b, d] = [a, b, d].map(|matrix| matrix.name(ctx));
    format!("{copy}{{ {operation} op; op.run(*{a}, *{b}, *{d}); }}\n")
}

/// Cell by cell, between fragments of one use: an operation's lanes hold the same cells of a
/// fragment whatever its element type, but not of another use's.
pub fn cast(ctx: &Context, op: &CastOp) -> String {
    let [input, output] = [op.input(ctx), op.output(ctx)].map(|matrix| matrix_ty(ctx, matrix));
    if (input.ident, input.shape) != (output.ident, output.shape) {
        return unsupported(
            ctx,
            format!(
                "a tensor-op fragment cast from {:?} to {:?}: an operation lays their cells out \
                 differently",
                input.ident, output.ident
            ),
        );
    }
    let elem = output.elem_ty.to_cpp(ctx);
    copy(op.input(ctx).name(ctx), op.output(ctx).name(ctx), &elem)
}

/// Every cell of `from` into `to`, through `convert` (a type, or nothing).
fn copy(from: impl Display, to: impl Display, convert: &str) -> String {
    cells(to, |cell| format!("{cell} = {convert}((*{from})[i]);"))
}
