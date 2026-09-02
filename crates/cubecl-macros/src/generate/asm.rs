use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

use crate::{
    parse::asm::{
        AsmArgs, AsmExpression, DirSpec, DualDirSpec, DualDirSpecExpression, FormatString,
        RegOperandBody, RegOperandKind, RegSpec,
    },
    paths::prelude_type,
    scope::Context,
};

impl AsmExpression {
    pub fn to_tokens(&self, ctx: &mut Context) -> TokenStream {
        let builder = prelude_type("BuildAsmExpand");

        let asm = &self.asm;
        let inputs = self
            .inputs
            .iter()
            .map(|(expr, spec)| {
                let expr = expr.to_tokens(ctx);
                let spec = match spec {
                    RegOperandKind::DirSpec(dir, reg) => quote![#dir, #reg],
                    RegOperandKind::DualDirSpec(dir, reg) => quote![#dir, #reg],
                };
                quote![#expr, #spec]
            })
            .collect::<Vec<_>>();
        let outputs = self
            .outputs
            .iter()
            .map(|(expr, spec)| {
                let expr = expr.to_tokens(ctx);
                quote![&#expr, #spec]
            })
            .collect::<Vec<_>>();
        let options = &self.options;

        quote! {{
            #builder::new(#asm)
            #(.push_input(scope, #inputs))*
            #(.push_output(scope, #outputs))*
            #(.#options())*
            .register(scope)
        }}
    }
}

impl ToTokens for RegSpec {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let reg_spec = prelude_type("RegSpec");
        tokens.extend(match self {
            RegSpec::Inferred(_) => quote![#reg_spec::Inferred],
            RegSpec::Class(ident) => {
                let name = ident.to_string();
                quote![#reg_spec::Class(#name.into())]
            }
            RegSpec::Explicit(name) => quote![#reg_spec::Explicit(#name.into())],
        });
    }
}

impl ToTokens for DirSpec {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let kind = prelude_type("InputKind");
        tokens.extend(match self {
            DirSpec::In(_) => quote![#kind::In],
            DirSpec::MemIn(_) => quote![#kind::MemIn],
            DirSpec::MemOut(_) => quote![#kind::MemOut],
            DirSpec::Lateout(_) | DirSpec::Out(_) => unreachable!(),
        });
    }
}

impl ToTokens for DualDirSpec {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let kind = prelude_type("InputKind");
        tokens.extend(match self {
            DualDirSpec::MemInout(_) => quote![#kind::MemInout],
            DualDirSpec::Inout(_) | DualDirSpec::Inlateout(_) => unreachable!(),
        });
    }
}

impl AsmArgs {
    pub fn generate_format_call(&self) -> syn::Result<TokenStream> {
        let registers = self.registers();
        let num_out = self.out_registers().count();

        let mut formats = self.formats.iter().map(|it| match &it.format {
            FormatString::Lit(lit_str) => lit_str.value(),
            FormatString::Macro(_) => unreachable!(),
        });

        // Results come before inputs in the IR, so the first input ID must be right after the last
        // output, even if that's not the case in code.
        let mut out_idx = 0;
        let mut in_idx = num_out;

        let fmt_str = formats.join("\n");
        let mut fmt_args = vec![];

        for reg in registers {
            match reg.body {
                RegOperandBody::DirSpec(dir_spec, ..) => {
                    let placeholder = match dir_spec {
                        DirSpec::In(_) | DirSpec::MemIn(_) | DirSpec::MemOut(_) => {
                            let placeholder = format!("${in_idx}");
                            in_idx += 1;
                            placeholder
                        }
                        DirSpec::Out(_) | DirSpec::Lateout(_) => {
                            let placeholder = format!("${out_idx}");
                            out_idx += 1;
                            placeholder
                        }
                    };
                    let fmt_arg = match reg.param_name {
                        Some(name) => quote![#name = #placeholder],
                        None => quote![#placeholder],
                    };
                    fmt_args.push(fmt_arg);
                }
                RegOperandBody::DualDirSpec(dir_spec, ..) => {
                    let placeholder = match dir_spec {
                        DualDirSpec::MemInout(_) => {
                            let placeholder = format!("${in_idx}");
                            in_idx += 1;
                            placeholder
                        }
                        DualDirSpec::Inout(_) | DualDirSpec::Inlateout(_) => {
                            panic!("inout params not yet supported")
                        }
                    };
                    let fmt_arg = match reg.param_name {
                        Some(name) => quote![#name = #placeholder],
                        None => quote![#placeholder],
                    };
                    fmt_args.push(fmt_arg);
                }
                RegOperandBody::Const(expr) => match reg.param_name {
                    Some(name) => fmt_args.push(quote![#name = #expr]),
                    None => fmt_args.push(quote![#expr]),
                },
                RegOperandBody::Sym(..) | RegOperandBody::Label(..) => unimplemented!(),
            }
        }

        Ok(quote![cubecl::__private::format!(#fmt_str, #(#fmt_args),*)])
    }
}

pub fn generate_asm_unexpanded(tokens: TokenStream) -> syn::Result<TokenStream> {
    let asm_spec: AsmArgs = syn::parse2(tokens)?;
    asm_spec.validate()?;

    let unexpanded_value = prelude_type("unexpanded_value");

    let registers = asm_spec.registers();

    let mut inputs = vec![];
    let mut outputs = vec![];

    for reg in registers {
        match reg.body {
            RegOperandBody::DirSpec(dir_spec, _, expr) => match dir_spec {
                DirSpec::In(_) | DirSpec::MemIn(_) | DirSpec::MemOut(_) => {
                    inputs.push(expr);
                }
                DirSpec::Out(_) | DirSpec::Lateout(_) => {
                    outputs.push(expr);
                }
            },
            RegOperandBody::DualDirSpec(dir_spec, _, DualDirSpecExpression::Single(expr)) => {
                match dir_spec {
                    DualDirSpec::MemInout(_) => {
                        inputs.push(expr);
                    }
                    DualDirSpec::Inout(_) | DualDirSpec::Inlateout(_) => {
                        unimplemented!("inout params not yet supported")
                    }
                }
            }
            RegOperandBody::DualDirSpec(..) => {
                unimplemented!("inout params not yet supported")
            }
            RegOperandBody::Const(_) => {}
            RegOperandBody::Sym(..) | RegOperandBody::Label(..) => unimplemented!(),
        }
    }

    let asm = asm_spec.generate_format_call()?;

    Ok(quote! {{
        let _ = #asm;
        #(let _ = #inputs;)*
        #(#outputs = #unexpanded_value();)*
    }})
}
