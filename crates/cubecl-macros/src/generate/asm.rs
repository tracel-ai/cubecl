use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    parse::asm::{AsmArgs, AsmExpression, DirSpec, FormatString, RegOperandBody},
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
            .map(|it| it.to_tokens(ctx))
            .collect::<Vec<_>>();
        let outputs = self
            .outputs
            .iter()
            .map(|it| it.to_tokens(ctx))
            .collect::<Vec<_>>();
        let options = &self.options;

        quote! {{
            #builder::new(#asm)
            #(.push_input(scope, #inputs))*
            #(.push_output(scope, &#outputs))*
            #(.#options())*
            .register(scope)
        }}
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
                        DirSpec::In(_) => {
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
                RegOperandBody::DualDirSpec(..) => {
                    panic!("inout params not yet supported")
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
                DirSpec::In(_) => {
                    inputs.push(expr);
                }
                DirSpec::Out(_) | DirSpec::Lateout(_) => {
                    outputs.push(expr);
                }
            },
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
