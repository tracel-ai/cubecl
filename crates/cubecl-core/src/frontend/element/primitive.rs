use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::{ConstantScalarValue, Elem, FloatKind, IntKind},
    new_ir::{
        Expand, Expanded, Expr, Expression, GlobalVariable, MaxExpr, MinExpr, SquareType,
        StaticExpand, StaticExpanded, UnaryOp, Vectorization,
    },
    prelude::{VecIndex, VecIndexMut},
    unexpanded, Runtime,
};
use cubecl_common::operator::Operator;
use half::{bf16, f16};
use num_traits::{NumAssign, NumCast, ToPrimitive};

use super::{ArgSettings, LaunchArg, LaunchArgExpand};

pub trait Numeric:
    Primitive
    + NumCast
    + NumAssign
    + PartialOrd
    + PartialEq
    + StaticExpand<Expanded: NumericExpandStatic>
    + VecIndex
    + VecIndexMut
    + Send
    + Sync
{
    fn new<N: ToPrimitive>(n: N) -> Self {
        <Self as NumCast>::from(n).unwrap()
    }
}
pub trait Float: Numeric + num_traits::Float {
    fn erf(self) -> Self {
        unexpanded!()
    }
}
pub trait Integer: Numeric + Ord {}

pub trait NumericExpandStatic: StaticExpanded + Sized
where
    Self::Unexpanded: Numeric,
{
    #[allow(clippy::new_ret_no_self)]
    fn new(n: impl ToPrimitive) -> impl Expr<Output = Self::Unexpanded> {
        <Self::Unexpanded as NumCast>::from(n).unwrap()
    }
}

pub trait IntegerExpand: Expanded<Unexpanded: Numeric> + Sized {
    fn min(
        self,
        other: impl Expr<Output = Self::Unexpanded>,
    ) -> impl Expr<Output = Self::Unexpanded> {
        MinExpr::new(self.inner(), other)
    }

    fn max(
        self,
        other: impl Expr<Output = Self::Unexpanded>,
    ) -> impl Expr<Output = Self::Unexpanded> {
        MaxExpr::new(self.inner(), other)
    }
}

impl<T: StaticExpanded> NumericExpandStatic for T where T::Unexpanded: Numeric {}
impl<T: Expanded> IntegerExpand for T where T::Unexpanded: Integer {}

pub trait FloatExpand: Expanded + Sized
where
    Self::Unexpanded: Float,
{
    fn cos(self) -> impl Expr<Output = Self::Unexpanded> {
        CosExpr::new(self.inner())
    }

    fn sqrt(self) -> impl Expr<Output = Self::Unexpanded> {
        SqrtExpr::new(self.inner())
    }

    fn erf(self) -> impl Expr<Output = Self::Unexpanded> {
        ErfExpr::new(self.inner())
    }
}

impl<T: Expanded> FloatExpand for T where T::Unexpanded: Float {}

pub trait Primitive: SquareType + Copy + 'static {
    fn value(&self) -> ConstantScalarValue;
}

impl<T: Primitive> Expr for T {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::Literal {
            value: self.value(),
            vectorization: self.vectorization(),
            ty: <T as SquareType>::ir_type(),
        }
    }

    fn vectorization(&self) -> Vectorization {
        self.vectorization()
    }
}

macro_rules! num_un_op {
    ($name:ident, $trait:path, $op:ident) => {
        pub struct $name<In: Expr>(pub UnaryOp<In, In::Output>)
        where
            In::Output: $trait;

        impl<In: Expr> $name<In>
        where
            In::Output: $trait,
        {
            pub fn new(input: In) -> Self {
                Self(UnaryOp::new(input))
            }
        }

        impl<In: Expr> Expr for $name<In>
        where
            In::Output: $trait,
        {
            type Output = In::Output;

            fn expression_untyped(&self) -> Expression {
                Expression::Unary {
                    input: Box::new(self.0.input.expression_untyped()),
                    operator: Operator::$op,
                    vectorization: self.vectorization(),
                    ty: In::Output::ir_type(),
                }
            }

            fn vectorization(&self) -> Vectorization {
                self.0.input.vectorization()
            }
        }
    };
}

num_un_op!(CosExpr, Float, Cos);
num_un_op!(SqrtExpr, Float, Sqrt);
num_un_op!(ErfExpr, Float, Erf);

macro_rules! primitive {
    ($primitive:ident, $var_type:expr) => {
        impl SquareType for $primitive {
            fn ir_type() -> Elem {
                $var_type
            }
        }
    };
}

macro_rules! numeric_primitive {
    ($primitive:ident, $var_type:expr, $expand_name:ident) => {
        primitive!($primitive, $var_type);

        pub struct $expand_name<Inner: Expr<Output = $primitive>>(Inner);
        impl Expand for $primitive {
            type Expanded<Inner: Expr<Output = Self>> = $expand_name<Inner>;

            fn expand<Inner: Expr<Output = Self>>(
                inner: Inner,
            ) -> <Self as Expand>::Expanded<Inner> {
                $expand_name(inner)
            }
        }
        impl StaticExpand for $primitive {
            type Expanded = $expand_name<Self>;
        }
        impl<Inner: Expr<Output = $primitive>> Expanded for $expand_name<Inner> {
            type Unexpanded = $primitive;

            fn inner(self) -> impl Expr<Output = Self::Unexpanded> {
                self.0
            }
        }

        impl Numeric for $primitive {}
        impl VecIndex for $primitive {}
        impl VecIndexMut for $primitive {}
    };
}

macro_rules! int_primitive {
    ($primitive:ident, $var_type:expr, $kind:expr, $expand_name:ident) => {
        numeric_primitive!($primitive, $var_type($kind), $expand_name);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> ConstantScalarValue {
                ConstantScalarValue::Int(*self as i64, $kind)
            }
        }
    };
}

macro_rules! uint_primitive {
    ($primitive:ident, $var_type:expr, $expand_name:ident) => {
        numeric_primitive!($primitive, $var_type, $expand_name);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> ConstantScalarValue {
                ConstantScalarValue::UInt(*self as u64)
            }
        }
    };
}

macro_rules! float_primitive {
    ($primitive:ident, $var_type:expr, $kind:expr, $expand_name:ident) => {
        numeric_primitive!($primitive, $var_type($kind), $expand_name);

        impl Float for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> ConstantScalarValue {
                ConstantScalarValue::Float(self.to_f64().unwrap(), $kind)
            }
        }
    };
}

int_primitive!(i32, Elem::Int, IntKind::I32, I32Expand);
int_primitive!(i64, Elem::Int, IntKind::I64, I64Expand);
uint_primitive!(u32, Elem::UInt, U32Expand);
float_primitive!(f16, Elem::Float, FloatKind::F16, F16Expand);
float_primitive!(bf16, Elem::Float, FloatKind::BF16, BF16Expand);
float_primitive!(f32, Elem::Float, FloatKind::F32, F32Expand);
float_primitive!(f64, Elem::Float, FloatKind::F64, F64Expand);
primitive!(bool, Elem::Bool);

impl Primitive for bool {
    fn value(&self) -> ConstantScalarValue {
        ConstantScalarValue::Bool(*self)
    }
}

/// Similar to [ArgSettings], however only for scalar types that don't depend on the [Runtime]
/// trait.
pub trait ScalarArgSettings: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>);
}

#[derive(new)]
pub struct ScalarArg<T: Numeric + ScalarArgSettings> {
    elem: T,
}

impl<T: Numeric + ScalarArgSettings, R: Runtime> ArgSettings<R> for ScalarArg<T> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        self.elem.register(launcher);
    }
}

impl<T: Numeric + ScalarArgSettings + LaunchArgExpand> LaunchArg for T {
    type RuntimeArg<'a, R: Runtime> = ScalarArg<T>;
}
impl<T: Numeric + ScalarArgSettings> LaunchArgExpand for T {
    fn expand(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self> {
        assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
        builder.scalar(T::ir_type())
    }
}

impl ScalarArgSettings for f16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f16(*self);
    }
}

impl ScalarArgSettings for bf16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_bf16(*self);
    }
}

impl ScalarArgSettings for f32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(*self);
    }
}

impl ScalarArgSettings for f64 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f64(*self);
    }
}

impl ScalarArgSettings for i32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i32(*self);
    }
}

impl ScalarArgSettings for i64 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i64(*self);
    }
}

impl ScalarArgSettings for u32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(*self);
    }
}
