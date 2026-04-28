use crate as cubecl;
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use cubecl::prelude::*;
use cubecl_ir::{
    Arithmetic, Bitwise, Comparison, Operation, Operator, Scope, StorageType, Type,
    features::ComplexUsage,
};
use cubecl_macros::intrinsic;

#[cube]
#[allow(unused_variables)]
/// Push a validation error that will make the kernel compilation to fail.
///
/// # Notes
///
/// The error can be caught after the kernel is launched.
pub fn push_validation_error(#[comptime] msg: String) {
    intrinsic! {|scope| scope.push_error(msg)}
}

fn collect_complex_storage_types(types: impl IntoIterator<Item = Type>) -> Vec<StorageType> {
    let mut storages = Vec::new();

    for ty in types {
        if ty.is_semantic() {
            continue;
        }

        let storage = ty.storage_type();
        if storage.elem_type().is_complex() && !storages.contains(&storage) {
            storages.push(storage);
        }
    }

    storages
}

fn require_complex_usage(
    scope: &mut Scope,
    types: impl IntoIterator<Item = Type>,
    usage: ComplexUsage,
    op_name: &'static str,
) {
    let storages = collect_complex_storage_types(types);
    if storages.is_empty() {
        return;
    }

    // `scope.properties` is populated for all production paths (kernel launch
    // and compute-builder scopes). When it is unset — only possible from
    // hand-rolled compiler tests that construct a bare `Scope` — we cannot
    // decide capability and fall through silently. Such tests must set
    // `device_properties` if they exercise complex types.
    let Some(properties) = scope.properties.clone() else {
        return;
    };

    for storage in storages {
        if !properties.supports_complex_usage(storage, usage) {
            scope.push_error(format!(
                "Complex operation `{op_name}` requires {usage:?} support for `{storage}`, but the active runtime does not advertise it."
            ));
        }
    }
}

fn reject_complex_operation(
    scope: &mut Scope,
    types: impl IntoIterator<Item = Type>,
    op_name: &'static str,
) {
    let storages = collect_complex_storage_types(types);
    if storages.is_empty() {
        return;
    }

    let supported = storages
        .into_iter()
        .map(|storage| storage.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    scope.push_error(format!(
        "Complex operation `{op_name}` is not part of the v1 complex contract for `{supported}`."
    ));
}

pub(crate) fn validate_complex_operation(scope: &mut Scope, operation: &Operation) {
    match operation {
        Operation::Arithmetic(arithmetic) => match arithmetic {
            Arithmetic::Add(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Core, "+")
            }
            Arithmetic::Sub(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Core, "-")
            }
            Arithmetic::Mul(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Core, "*")
            }
            Arithmetic::Div(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Core, "/")
            }
            Arithmetic::Neg(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Core, "neg")
            }
            Arithmetic::Conj(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Core, "conj")
            }
            Arithmetic::Abs(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "abs")
            }
            Arithmetic::Exp(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "exp")
            }
            Arithmetic::Log(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "log")
            }
            Arithmetic::Sin(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "sin")
            }
            Arithmetic::Cos(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "cos")
            }
            Arithmetic::Sqrt(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "sqrt")
            }
            Arithmetic::Tanh(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Math, "tanh")
            }
            Arithmetic::Powf(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Math, "powf")
            }
            Arithmetic::Fma(op) => {
                reject_complex_operation(scope, [op.a.ty, op.b.ty, op.c.ty], "fma")
            }
            Arithmetic::SaturatingAdd(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "saturating_add")
            }
            Arithmetic::SaturatingSub(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "saturating_sub")
            }
            Arithmetic::Log1p(op) => reject_complex_operation(scope, [op.input.ty], "log1p"),
            Arithmetic::Expm1(op) => reject_complex_operation(scope, [op.input.ty], "exp_m1"),
            Arithmetic::Tan(op) => reject_complex_operation(scope, [op.input.ty], "tan"),
            Arithmetic::Sinh(op) => reject_complex_operation(scope, [op.input.ty], "sinh"),
            Arithmetic::Cosh(op) => reject_complex_operation(scope, [op.input.ty], "cosh"),
            Arithmetic::ArcCos(op) => reject_complex_operation(scope, [op.input.ty], "acos"),
            Arithmetic::ArcSin(op) => reject_complex_operation(scope, [op.input.ty], "asin"),
            Arithmetic::ArcTan(op) => reject_complex_operation(scope, [op.input.ty], "atan"),
            Arithmetic::ArcSinh(op) => reject_complex_operation(scope, [op.input.ty], "asinh"),
            Arithmetic::ArcCosh(op) => reject_complex_operation(scope, [op.input.ty], "acosh"),
            Arithmetic::ArcTanh(op) => reject_complex_operation(scope, [op.input.ty], "atanh"),
            Arithmetic::Degrees(op) => reject_complex_operation(scope, [op.input.ty], "to_degrees"),
            Arithmetic::Radians(op) => reject_complex_operation(scope, [op.input.ty], "to_radians"),
            Arithmetic::ArcTan2(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "atan2")
            }
            Arithmetic::Powi(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "powi"),
            Arithmetic::Hypot(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "hypot")
            }
            Arithmetic::Rhypot(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "rhypot")
            }
            Arithmetic::InverseSqrt(op) => {
                reject_complex_operation(scope, [op.input.ty], "inverse_sqrt")
            }
            Arithmetic::Round(op) => reject_complex_operation(scope, [op.input.ty], "round"),
            Arithmetic::Floor(op) => reject_complex_operation(scope, [op.input.ty], "floor"),
            Arithmetic::Ceil(op) => reject_complex_operation(scope, [op.input.ty], "ceil"),
            Arithmetic::Trunc(op) => reject_complex_operation(scope, [op.input.ty], "trunc"),
            Arithmetic::Erf(op) => reject_complex_operation(scope, [op.input.ty], "erf"),
            Arithmetic::Recip(op) => reject_complex_operation(scope, [op.input.ty], "recip"),
            Arithmetic::Clamp(op) => reject_complex_operation(
                scope,
                [op.input.ty, op.min_value.ty, op.max_value.ty],
                "clamp",
            ),
            Arithmetic::Modulo(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "%"),
            Arithmetic::Max(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "max"),
            Arithmetic::Min(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "min"),
            Arithmetic::Remainder(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "remainder")
            }
            Arithmetic::Magnitude(op) => {
                reject_complex_operation(scope, [op.input.ty], "magnitude")
            }
            Arithmetic::Normalize(op) => {
                reject_complex_operation(scope, [op.input.ty], "normalize")
            }
            Arithmetic::Dot(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "dot"),
            Arithmetic::MulHi(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "mul_hi")
            }
            Arithmetic::VectorSum(op) => {
                reject_complex_operation(scope, [op.input.ty], "vector_sum")
            }
        },
        Operation::Comparison(comparison) => match comparison {
            Comparison::Equal(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Compare, "==")
            }
            Comparison::NotEqual(op) => {
                require_complex_usage(scope, [op.lhs.ty, op.rhs.ty], ComplexUsage::Compare, "!=")
            }
            Comparison::Lower(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "<"),
            Comparison::LowerEqual(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "<=")
            }
            Comparison::GreaterEqual(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], ">=")
            }
            Comparison::Greater(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], ">"),
            Comparison::IsNan(op) => reject_complex_operation(scope, [op.input.ty], "is_nan"),
            Comparison::IsInf(op) => reject_complex_operation(scope, [op.input.ty], "is_inf"),
        },
        Operation::Bitwise(bitwise) => match bitwise {
            Bitwise::BitwiseAnd(op)
            | Bitwise::BitwiseOr(op)
            | Bitwise::BitwiseXor(op)
            | Bitwise::ShiftLeft(op)
            | Bitwise::ShiftRight(op) => {
                reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "bitwise")
            }
            Bitwise::CountOnes(op)
            | Bitwise::ReverseBits(op)
            | Bitwise::BitwiseNot(op)
            | Bitwise::LeadingZeros(op)
            | Bitwise::TrailingZeros(op)
            | Bitwise::FindFirstSet(op) => {
                reject_complex_operation(scope, [op.input.ty], "bitwise")
            }
        },
        Operation::Operator(operator) => match operator {
            Operator::Real(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Core, "real_val")
            }
            Operator::Imag(op) => {
                require_complex_usage(scope, [op.input.ty], ComplexUsage::Core, "imag_val")
            }
            Operator::And(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "&&"),
            Operator::Or(op) => reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "||"),
            Operator::Not(op) => reject_complex_operation(scope, [op.input.ty], "!"),
            _ => {}
        },
        _ => {}
    }
}

pub(crate) fn validate_complex_assign_operation(scope: &mut Scope, operation: &Operation) {
    match operation {
        Operation::Arithmetic(Arithmetic::Add(op)) => {
            reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "+=")
        }
        Operation::Arithmetic(Arithmetic::Sub(op)) => {
            reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "-=")
        }
        Operation::Arithmetic(Arithmetic::Mul(op)) => {
            reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "*=")
        }
        Operation::Arithmetic(Arithmetic::Div(op)) => {
            reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "/=")
        }
        Operation::Arithmetic(Arithmetic::Modulo(op)) => {
            reject_complex_operation(scope, [op.lhs.ty, op.rhs.ty], "%=")
        }
        _ => {}
    }
}
