use crate::tracker::VariableTracker;

use super::{
    base::{Codegen, CodegenKind},
    expr::codegen_expr,
};

/// Codegen for binary operations (+, -, *, etc.)
pub(crate) fn codegen_binary(
    binary: &syn::ExprBinary,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> Codegen {
    let lhs = codegen_expr(&binary.left, loop_level, variable_tracker);
    let (lhs, kind_lhs, lhs_array) = lhs.process();
    let (rhs, kind_rhs, _) = codegen_expr(&binary.right, loop_level, variable_tracker).process();

    if matches!(kind_lhs, CodegenKind::Comptime) && matches!(kind_rhs, CodegenKind::Comptime) {
        return Codegen::new(
            quote::quote! {
                #binary
            },
            CodegenKind::Comptime,
        );
    }

    Codegen::new(
        match binary.op {
            syn::BinOp::Add(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::add::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Sub(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::sub::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Mul(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::mul::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Div(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::div::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Rem(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::rem::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Ne(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::ne::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Gt(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::gt::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Ge(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::ge::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Lt(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::lt::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Le(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::le::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Eq(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::eq::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::AddAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            cubecl::frontend::add_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            cubecl::frontend::add_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::SubAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            cubecl::frontend::sub_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            cubecl::frontend::sub_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::MulAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            cubecl::frontend::mul_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            cubecl::frontend::mul_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::DivAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            cubecl::frontend::div_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            cubecl::frontend::div_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::And(_) => quote::quote! {
                {

                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::and::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Or(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::or::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::BitAnd(_) => quote::quote! {
                {

                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::bitand::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::BitXor(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::bitxor::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Shl(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::shl::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Shr(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    cubecl::frontend::shr::expand(context, _lhs, _rhs)
                }
            },
            _ => todo!("Codegen: unsupported op {:?}", binary.op),
        },
        CodegenKind::Expand,
    )
}

/// Codegen for unary operations
pub(crate) fn codegen_unary(
    unary: &syn::ExprUnary,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> Codegen {
    let (inner, kind, _) = codegen_expr(&unary.expr, loop_level, variable_tracker).process();

    if matches!(kind, CodegenKind::Comptime) {
        return Codegen::new(
            quote::quote! {
                #unary
            },
            CodegenKind::Comptime,
        );
    }

    Codegen::new(
        match unary.op {
            syn::UnOp::Not(_) => quote::quote! {
                {
                    let _inner = #inner;
                    cubecl::frontend::not::expand(context, _inner)
                }
            },
            syn::UnOp::Deref(_) => inner,
            _ => todo!("Codegen: unsupported op {:?}", unary.op),
        },
        CodegenKind::Expand,
    )
}
