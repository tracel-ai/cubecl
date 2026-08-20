use core::ops::{Add, Div, Mul, Neg, Sub};

use cubecl_ir::{
    ComplexKind, ConstantValue, ElemType, ExpandValue, FloatKind, Scope,
    dialect::{
        cmp::{FEqualOp, FNotEqualOp},
        math::*,
    },
    features::ComplexUsage,
    interfaces::TypedExt,
    pliron::{
        builtin::op_interfaces::OneResultInterface, context::Context, op::Op, r#type::TypeHandle,
        value::Value,
    },
    types::{
        VectorType,
        scalar::{Complex32Type, Complex64Type},
    },
};
use cubecl_runtime::client::ComputeClient;

use crate::{
    Runtime,
    frontend::{
        Abs, AbsNativeExpand, Cos, CosNativeExpand, Exp, ExpNativeExpand, Log, LogNativeExpand,
        Powf, PowfNativeExpand, ScalarArgSettings, Sin, SinNativeExpand, Sqrt, SqrtNativeExpand,
        Tanh, TanhNativeExpand,
        operation::{
            AddNativeExpand, DivNativeExpand, MulNativeExpand, NegNativeExpand,
            PartialEqNativeExpand, SubNativeExpand, binary_expand, unary_expand,
        },
        require_complex_usage,
    },
    prelude::{
        CubeDebug, CubePrimitive, CubeType, IntoExpand, IntoRuntime, KernelBuilder, KernelLauncher,
        LaunchArg, NativeAssign, NativeExpand, Scalar, impl_scalar_launch,
    },
    unexpanded,
};

pub trait ComplexCore:
    Scalar
    + IntoRuntime
    + CubePrimitive<
        Scalar: ComplexNativeExpand<FloatElem = Self::FloatElem>
                    + AddNativeExpand
                    + SubNativeExpand
                    + MulNativeExpand
                    + DivNativeExpand
                    + NegNativeExpand
                    + PartialEqNativeExpand,
    > + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Copy
    + Clone
    + PartialEq
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    type FloatElem: Scalar;

    fn conj(self) -> Self {
        unexpanded!()
    }

    fn real_val(self) -> Self::FloatElem {
        unexpanded!()
    }

    fn imag_val(self) -> Self::FloatElem {
        unexpanded!()
    }

    fn supported_complex_uses<R: Runtime>(
        client: &ComputeClient<R>,
    ) -> enumset::EnumSet<ComplexUsage> {
        client.properties().complex_usage(Self::elem_type_native())
    }
}

pub trait ComplexCompare: ComplexCore {}

pub trait ComplexMath:
    ComplexCore + Abs<AbsElem = Self::FloatElem> + Exp + Log + Sin + Cos + Sqrt + Tanh + Powf
{
}

pub trait ComplexNativeExpand {
    type FloatElem: Scalar;
    fn __expand_native_conj(scope: &Scope, input: ExpandValue) -> ExpandValue;
    fn __expand_native_real(scope: &Scope, input: ExpandValue) -> ExpandValue;
    fn __expand_native_imag(scope: &Scope, input: ExpandValue) -> ExpandValue;
}

pub trait ComplexCoreExpand {
    type FloatElem: Scalar;
    fn __expand_conj_method(self, scope: &Scope) -> Self;
    fn __expand_real_val_method(self, scope: &Scope) -> NativeExpand<Self::FloatElem>;
    fn __expand_imag_val_method(self, scope: &Scope) -> NativeExpand<Self::FloatElem>;
}

impl<T: ComplexCore> ComplexCoreExpand for NativeExpand<T> {
    type FloatElem = T::FloatElem;

    fn __expand_conj_method(self, scope: &Scope) -> Self {
        T::Scalar::__expand_native_conj(scope, self.into()).into()
    }

    fn __expand_real_val_method(self, scope: &Scope) -> NativeExpand<T::FloatElem> {
        T::Scalar::__expand_native_real(scope, self.into()).into()
    }

    fn __expand_imag_val_method(self, scope: &Scope) -> NativeExpand<T::FloatElem> {
        T::Scalar::__expand_native_imag(scope, self.into()).into()
    }
}

fn complex_component<O>(
    scope: &Scope,
    input: ExpandValue,
    out_scalar: TypeHandle,
    op: impl FnOnce(&mut Context, Value, TypeHandle) -> O,
) -> ExpandValue
where
    O: Op + OneResultInterface,
{
    let input = input.read_value(scope);
    let vector_size = input.vector_size(scope.ctx());
    let out_ty = if vector_size == 1 {
        out_scalar
    } else {
        VectorType::get(scope.ctx(), out_scalar, vector_size).into()
    };
    let operation = op(scope.ctx_mut(), input, out_ty);
    scope.register_with_result(&operation).into()
}

macro_rules! impl_complex_unary {
    ($primitive:ty, $trait:ident, $native:ident, $method:ident, $op:ty, $name:literal) => {
        impl $trait for $primitive {}
        impl $native for $primitive {
            fn $method(scope: &Scope, input: ExpandValue) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Math, $name);
                unary_expand(scope, input, <$op>::new)
            }
        }
    };
}

macro_rules! impl_complex {
    ($primitive:ty, $kind:ident, $float:ty, $ir_ty:ty, $float_kind:ident) => {
        impl CubeType for $primitive {
            type ExpandType = NativeExpand<Self>;
        }

        impl CubeDebug for $primitive {}

        impl Scalar for $primitive {
            fn elem_type_native() -> ElemType {
                ElemType::Complex(ComplexKind::$kind)
            }
        }

        impl CubePrimitive for $primitive {
            type Scalar = Self;
            type Size = crate::prelude::Const<1>;
            type WithScalar<S: Scalar> = S;

            fn from_const_value(value: ConstantValue) -> Self {
                let ConstantValue::Complex(re, im) = value else {
                    unreachable!("expected complex constant")
                };
                <$primitive>::new(re as $float, im as $float)
            }

            fn __expand_as_type(scope: &Scope) -> TypeHandle {
                <$ir_ty>::get(scope.ctx()).into()
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
                self.into()
            }
        }

        impl IntoExpand for $primitive {
            type Expand = NativeExpand<Self>;
            fn into_expand(self, _scope: &Scope) -> Self::Expand {
                self.into()
            }
        }

        impl NativeAssign for $primitive {}
        impl_scalar_launch!($primitive);

        impl ComplexNativeExpand for $primitive {
            type FloatElem = $float;

            fn __expand_native_conj(scope: &Scope, input: ExpandValue) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Core, "conj");
                unary_expand(scope, input, CConjOp::new)
            }

            fn __expand_native_real(scope: &Scope, input: ExpandValue) -> ExpandValue {
                require_complex_usage(
                    scope,
                    Self::elem_type_native(),
                    ComplexUsage::Core,
                    "real_val",
                );
                complex_component(
                    scope,
                    input,
                    FloatKind::$float_kind.to_type(scope.ctx()),
                    |ctx, input, ty| CRealOp::new(ctx, ty, input),
                )
            }

            fn __expand_native_imag(scope: &Scope, input: ExpandValue) -> ExpandValue {
                require_complex_usage(
                    scope,
                    Self::elem_type_native(),
                    ComplexUsage::Core,
                    "imag_val",
                );
                complex_component(
                    scope,
                    input,
                    FloatKind::$float_kind.to_type(scope.ctx()),
                    |ctx, input, ty| CImagOp::new(ctx, ty, input),
                )
            }
        }

        impl ComplexCore for $primitive {
            type FloatElem = $float;
        }
        impl ComplexCompare for $primitive {}
        impl ComplexMath for $primitive {}

        impl AddNativeExpand for $primitive {
            fn __expand_native_add(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Core, "+");
                binary_expand(scope, lhs, rhs, FAddOp::new)
            }
        }
        impl SubNativeExpand for $primitive {
            fn __expand_native_sub(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Core, "-");
                binary_expand(scope, lhs, rhs, FSubOp::new)
            }
        }
        impl MulNativeExpand for $primitive {
            fn __expand_native_mul(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Core, "*");
                binary_expand(scope, lhs, rhs, FMulOp::new)
            }
        }
        impl DivNativeExpand for $primitive {
            fn __expand_native_div(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Core, "/");
                binary_expand(scope, lhs, rhs, FDivOp::new)
            }
        }
        impl NegNativeExpand for $primitive {
            fn __expand_native_neg(scope: &Scope, input: ExpandValue) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Core, "neg");
                unary_expand(scope, input, FNegOp::new)
            }
        }
        impl PartialEqNativeExpand for $primitive {
            fn __expand_native_eq(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Compare, "==");
                binary_expand(scope, lhs, rhs, FEqualOp::new)
            }
            fn __expand_native_ne(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Compare, "!=");
                binary_expand(scope, lhs, rhs, FNotEqualOp::new)
            }
        }

        impl Abs for $primitive {
            type AbsElem = $float;
        }
        impl AbsNativeExpand for $primitive {
            type AbsElem = $float;
            fn __expand_native_abs(scope: &Scope, input: ExpandValue) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Math, "abs");
                complex_component(
                    scope,
                    input,
                    FloatKind::$float_kind.to_type(scope.ctx()),
                    |ctx, input, ty| CAbsOp::new(ctx, ty, input),
                )
            }
        }

        impl_complex_unary!(
            $primitive,
            Exp,
            ExpNativeExpand,
            __expand_native_exp,
            ExpOp,
            "exp"
        );
        impl_complex_unary!(
            $primitive,
            Log,
            LogNativeExpand,
            __expand_native_ln,
            LogOp,
            "log"
        );
        impl_complex_unary!(
            $primitive,
            Sin,
            SinNativeExpand,
            __expand_native_sin,
            SinOp,
            "sin"
        );
        impl_complex_unary!(
            $primitive,
            Cos,
            CosNativeExpand,
            __expand_native_cos,
            CosOp,
            "cos"
        );
        impl_complex_unary!(
            $primitive,
            Sqrt,
            SqrtNativeExpand,
            __expand_native_sqrt,
            SqrtOp,
            "sqrt"
        );
        impl_complex_unary!(
            $primitive,
            Tanh,
            TanhNativeExpand,
            __expand_native_tanh,
            TanhOp,
            "tanh"
        );

        impl Powf for $primitive {}
        impl PowfNativeExpand for $primitive {
            fn __expand_native_powf(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                require_complex_usage(scope, Self::elem_type_native(), ComplexUsage::Math, "powf");
                binary_expand(scope, lhs, rhs, PowfOp::new)
            }
        }
    };
}

impl_complex!(num_complex::Complex<f32>, C32, f32, Complex32Type, F32);
impl_complex!(num_complex::Complex<f64>, C64, f64, Complex64Type, F64);
