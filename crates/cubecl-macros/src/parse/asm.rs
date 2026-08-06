//! Parse the same syntax as the compiler built-in `asm!` macro, so we can generate an assembly spec
//! based on it. The syntax is too complex for a simple declarative macro or 1 to 1 translation.
//! We parse all parts of the syntax to provide proper error messages for unsupported features.

use derive_more::Display;
use proc_macro2::{Span, TokenStream};
use quote::format_ident;
use syn::{
    Attribute, Expr, Ident, LitStr, Macro, Path, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned as _,
    token,
};

use crate::{expression::Expression, scope::Context};

mod kw {
    syn::custom_keyword!(clobber_abi);
    syn::custom_keyword!(options);

    syn::custom_keyword!(pure);
    syn::custom_keyword!(nomem);
    syn::custom_keyword!(readonly);
    syn::custom_keyword!(preserves_flags);
    syn::custom_keyword!(noreturn);
    syn::custom_keyword!(nostack);
    syn::custom_keyword!(raw);

    syn::custom_keyword!(out);
    syn::custom_keyword!(lateout);
    syn::custom_keyword!(inout);
    syn::custom_keyword!(inlateout);
    syn::custom_keyword!(sym);
    syn::custom_keyword!(label);
}

macro_rules! peek_select {
    ($input: expr, { $first_arm: path => $first_body: expr, $($arm:path => $body:expr),*; _ => $default: expr, }) => {
        if $input.peek($first_arm) {
            $first_body
        }
        $(else if $input.peek($arm) {
            $body
        })*
        else {
            $default
        }
    };
}

pub struct AsmArgs {
    pub formats: Vec<AsmAttrFormatString>,
    pub operands: Vec<AsmAttrOperand>,
}

pub struct AsmAttrFormatString {
    pub attrs: Vec<Attribute>,
    pub format: FormatString,
}

pub enum FormatString {
    Lit(LitStr),
    Macro(Macro),
}

#[derive(Clone)]
pub enum AsmOperand {
    ClobberAbi(ClobberAbi),
    AsmOptions(AsmOptions),
    RegOperand(RegOperand),
}

pub struct AsmAttrOperand {
    pub attrs: Vec<Attribute>,
    pub operand: AsmOperand,
}

#[derive(Clone)]
pub struct ClobberAbi {
    pub token: kw::clobber_abi,
    pub _abis: Punctuated<LitStr, Token![,]>,
}

#[derive(Clone)]
pub struct AsmOptions {
    pub options: Punctuated<AsmOption, Token![,]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Display)]
pub enum AsmOption {
    #[display("pure")]
    Pure(kw::pure),
    #[display("nomem")]
    Nomem(kw::nomem),
    #[display("readonly")]
    Readonly(kw::readonly),
    #[display("preserves_flags")]
    PreservesFlags(kw::preserves_flags),
    #[display("noreturn")]
    Noreturn(kw::noreturn),
    #[display("nostack")]
    Nostack(kw::nostack),
    #[display("raw")]
    Raw(kw::raw),
}

impl AsmOption {
    pub fn span(&self) -> Span {
        match self {
            AsmOption::Pure(kw) => kw.span(),
            AsmOption::Nomem(kw) => kw.span(),
            AsmOption::Readonly(kw) => kw.span(),
            AsmOption::PreservesFlags(kw) => kw.span(),
            AsmOption::Noreturn(kw) => kw.span(),
            AsmOption::Nostack(kw) => kw.span(),
            AsmOption::Raw(kw) => kw.span(),
        }
    }
}

#[derive(Clone)]
pub struct RegOperand {
    pub param_name: Option<Ident>,
    pub body: RegOperandBody,
}

#[derive(Clone)]
#[allow(unused, reason = "some fields not supported, but want to parse it")]
pub enum RegOperandBody {
    DirSpec(DirSpec, RegSpec, Expr),
    DualDirSpec(DualDirSpec, RegSpec, DualDirSpecExpression),
    Sym(kw::sym, Path),
    Const(Expr),
    Label(kw::label, syn::Block),
}

#[derive(Clone)]
#[allow(unused, reason = "not supported, but want to parse it")]
pub enum DualDirSpecExpression {
    Single(Expr),
    Pair(Expr, Expr),
}

#[derive(Clone)]
#[allow(unused, reason = "not supported, but want to parse it")]
pub enum RegSpec {
    Class(Ident),
    Inferred(Token![_]),
    Explicit(LitStr),
}

#[derive(Clone, Copy, Debug)]
#[allow(unused, reason = "parsing")]
pub enum DirSpec {
    In(token::In),
    Out(kw::out),
    Lateout(kw::lateout),
}

#[derive(Clone, Copy, Debug, Display)]
pub enum DualDirSpec {
    #[display("inout")]
    Inout(kw::inout),
    #[display("inlateout")]
    Inlateout(kw::inlateout),
}

impl DualDirSpec {
    pub fn span(&self) -> Span {
        match self {
            DualDirSpec::Inout(inout) => inout.span(),
            DualDirSpec::Inlateout(inlateout) => inlateout.span(),
        }
    }
}

// AsmArgs → AsmAttrFormatString ( , AsmAttrFormatString )* ( , AsmAttrOperand )* ,?
impl Parse for AsmArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut formats = Vec::new();
        let mut operands = Vec::new();

        formats.push(input.parse::<AsmAttrFormatString>()?);

        // Additional format strings
        while input.peek(Token![,]) {
            let fork = input.fork();
            fork.parse::<Token![,]>()?;
            if fork.parse::<AsmAttrFormatString>().is_ok() {
                input.parse::<Token![,]>()?;
                formats.push(input.parse()?);
            } else {
                break;
            }
        }

        // Operands
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            operands.push(input.parse()?);
        }

        Ok(AsmArgs { formats, operands })
    }
}

// AsmAttrFormatString → ( OuterAttribute )* FormatString
impl Parse for AsmAttrFormatString {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let format = input.parse()?;
        Ok(Self { attrs, format })
    }
}

// FormatString → STRING_LITERAL | RAW_STRING_LITERAL | MacroInvocation
impl Parse for FormatString {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(LitStr) {
            Ok(FormatString::Lit(input.parse()?))
        } else {
            Ok(FormatString::Macro(input.parse()?))
        }
    }
}

// AsmAttrOperand → ( OuterAttribute )* AsmOperand
impl Parse for AsmAttrOperand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let operand = input.parse()?;
        Ok(Self { attrs, operand })
    }
}

// AsmOperand → ClobberAbi | AsmOptions | RegOperand
impl Parse for AsmOperand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        peek_select!(input, {
            kw::clobber_abi => input.parse().map(AsmOperand::ClobberAbi),
            kw::options => input.parse().map(AsmOperand::AsmOptions);
            _ => input.parse().map(AsmOperand::RegOperand),
        })
    }
}

// ClobberAbi → clobber_abi ( Abi ( , Abi )* ,? )
impl Parse for ClobberAbi {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let token = kw::clobber_abi::parse(input)?;
        let content;
        syn::parenthesized!(content in input);
        Ok(Self {
            token,
            _abis: Punctuated::parse_terminated(&content)?,
        })
    }
}

// AsmOptions → options ( ( AsmOption ( , AsmOption )* ,? )? )
impl Parse for AsmOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        kw::options::parse(input)?;
        let content;
        syn::parenthesized!(content in input);
        let options = if content.is_empty() {
            Punctuated::new()
        } else {
            Punctuated::parse_terminated(&content)?
        };
        Ok(Self { options })
    }
}

// AsmOption → pure | nomem | readonly | preserves_flags | noreturn | nostack | att_syntax | raw
impl Parse for AsmOption {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        peek_select!(input, {
            kw::pure => Ok(AsmOption::Pure(input.parse()?)),
            kw::nomem => Ok(AsmOption::Nomem(input.parse()?)),
            kw::readonly => Ok(AsmOption::Readonly(input.parse()?)),
            kw::preserves_flags => Ok(AsmOption::PreservesFlags(input.parse()?)),
            kw::noreturn => Ok(AsmOption::Noreturn(input.parse()?)),
            kw::nostack => Ok(AsmOption::Nostack(input.parse()?)),
            kw::raw => Ok(AsmOption::Raw(input.parse()?));
            _ => {
                Err(syn::Error::new(input.span(), "expected asm option"))
            },
        })
    }
}

// RegOperand → ( ParamName = )? ( ... )
impl Parse for RegOperand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // NOTE: extend the peek here if you need keyword parameter names (e.g. `r#in`)
        let param_name = if input.peek(Ident) && input.peek2(Token![=]) {
            let name = input.parse::<Ident>()?;
            input.parse::<Token![=]>()?;
            Some(name)
        } else {
            None
        };
        let body = input.parse()?;
        Ok(Self { param_name, body })
    }
}

// RegOperand body alternatives
impl Parse for RegOperandBody {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        peek_select!(input, {
            token::Const => {
                input.parse::<Token![const]>()?;
                Ok(RegOperandBody::Const(input.parse()?))
            },
            token::In => {
                let dir = input.parse()?;
                let content;
                syn::parenthesized!(content in input);
                Ok(RegOperandBody::DirSpec(
                    dir,
                    content.parse()?,
                    input.parse()?,
                ))
            },
            kw::out => {
                let dir = input.parse()?;
                let content;
                syn::parenthesized!(content in input);
                Ok(RegOperandBody::DirSpec(
                    dir,
                    content.parse()?,
                    input.parse()?,
                ))
            },
            kw::lateout => {
                let dir = input.parse()?;
                let content;
                syn::parenthesized!(content in input);
                Ok(RegOperandBody::DirSpec(
                    dir,
                    content.parse()?,
                    input.parse()?,
                ))
            },
            kw::inout => {
                let dir = input.parse()?;
                let content;
                syn::parenthesized!(content in input);
                 Ok(RegOperandBody::DualDirSpec(
                    dir,
                    content.parse()?,
                    input.parse()?,
                ))
            },
            kw::inlateout => {
                let dir = input.parse()?;
                let content;
                syn::parenthesized!(content in input);
                 Ok(RegOperandBody::DualDirSpec(
                    dir,
                    content.parse()?,
                    input.parse()?,
                ))
            },
            kw::sym => Ok(RegOperandBody::Sym(input.parse()?, input.parse()?)),
            kw::label => Ok(RegOperandBody::Label(input.parse()?, input.parse()?));
            _ => Err(syn::Error::new(
                input.span(),
                "expected register operand directive",
            )),
        })
    }
}

// DualDirSpecExpression → Expression | Expression => Expression
impl Parse for DualDirSpecExpression {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let left: Expr = input.parse()?;
        if input.peek(Token![=>]) {
            input.parse::<Token![=>]>()?;
            let right: Expr = input.parse()?;
            Ok(DualDirSpecExpression::Pair(left, right))
        } else {
            Ok(DualDirSpecExpression::Single(left))
        }
    }
}

// RegSpec → RegisterClass | ExplicitRegister
impl Parse for RegSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(LitStr) {
            Ok(RegSpec::Explicit(input.parse()?))
        } else if input.peek(Token![_]) {
            Ok(RegSpec::Inferred(input.parse()?))
        } else {
            Ok(RegSpec::Class(input.parse()?))
        }
    }
}

// DirSpec → in | out | lateout
impl Parse for DirSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        peek_select!(input, {
            token::In => input.parse().map(DirSpec::In),
            kw::out => input.parse().map(DirSpec::Out),
            kw::lateout => input.parse().map(DirSpec::Lateout);
            _ => Err(syn::Error::new(
                input.span(),
                "expected `in`, `out`, or `lateout`",
            )),
        })
    }
}

// DualDirSpec → inout | inlateout
impl Parse for DualDirSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        peek_select!(input, {
            kw::inout => input.parse().map(DualDirSpec::Inout),
            kw::inlateout => input.parse().map(DualDirSpec::Inlateout);
            _ => Err(syn::Error::new(
                input.span(),
                "expected `inout` or `inlateout`",
            )),
        })
    }
}

impl AsmArgs {
    pub fn validate(&self) -> syn::Result<()> {
        for format in &self.formats {
            if !format.attrs.is_empty() {
                Err(syn::Error::new_spanned(
                    &format.attrs[0],
                    "Attributes in `gpu_asm` macro are not yet supported",
                ))?;
            }
            if let FormatString::Macro(mac) = &format.format {
                Err(syn::Error::new_spanned(
                    mac,
                    "Macro format strings are not supported",
                ))?;
            }
        }
        for operand in &self.operands {
            if !operand.attrs.is_empty() {
                Err(syn::Error::new_spanned(
                    &operand.attrs[0],
                    "Attributes in `gpu_asm` macro are not yet supported",
                ))?;
            }
            match &operand.operand {
                AsmOperand::ClobberAbi(clobber_abi) => {
                    Err(syn::Error::new(
                        clobber_abi.token.span(),
                        "`clobber_abi` is not supported for GPU assembly",
                    ))?;
                }
                AsmOperand::AsmOptions(options) => options.validate(self)?,
                AsmOperand::RegOperand(reg) => reg.validate()?,
            }
        }
        Ok(())
    }

    pub fn registers(&self) -> Vec<RegOperand> {
        self.operands
            .iter()
            .filter_map(|opd| match &opd.operand {
                AsmOperand::RegOperand(reg) => Some(reg.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn out_registers(&self) -> impl Iterator<Item = RegOperand> {
        self.registers().into_iter().filter(|it| match &it.body {
            RegOperandBody::DirSpec(dir, _, _) => {
                matches!(dir, DirSpec::Out(_) | DirSpec::Lateout(_))
            }
            RegOperandBody::DualDirSpec(..) => true,
            _ => false,
        })
    }
}

impl AsmOptions {
    pub fn validate(&self, args: &AsmArgs) -> syn::Result<()> {
        for option in &self.options {
            match option {
                AsmOption::Pure(token) => {
                    let has_readonly = self
                        .options
                        .iter()
                        .any(|it| matches!(it, AsmOption::Readonly(_)));
                    let has_nomem = self
                        .options
                        .iter()
                        .any(|it| matches!(it, AsmOption::Nomem(_)));
                    if !has_readonly && !has_nomem {
                        Err(syn::Error::new(
                            token.span(),
                            "`pure` must be combined with either `readonly` or `nomem`",
                        ))?;
                    }
                    if args.out_registers().count() == 0 {
                        Err(syn::Error::new(
                            token.span(),
                            "`pure` must have at least one out register",
                        ))?;
                    }
                }
                AsmOption::Nomem(_) | AsmOption::Readonly(_) => {}
                unsupported @ (AsmOption::PreservesFlags(_)
                | AsmOption::Noreturn(_)
                | AsmOption::Nostack(_)
                | AsmOption::Raw(_)) => Err(syn::Error::new(
                    option.span(),
                    format!("`{unsupported}` option is currently not supported"),
                ))?,
            }
        }
        Ok(())
    }
}

impl RegOperand {
    pub fn validate(&self) -> syn::Result<()> {
        match &self.body {
            RegOperandBody::DirSpec(dir, ..) => {
                // Maybe validate register class, leave for now since it's ignored. I wish there was
                // `compile_warning!()`...
                if let DirSpec::Lateout(token) = dir {
                    Err(syn::Error::new_spanned(
                        token,
                        "`lateout` direction is currently not supported",
                    ))?;
                }
            }
            RegOperandBody::DualDirSpec(dir, ..) => {
                Err(syn::Error::new(
                    dir.span(),
                    format!("`{}` direction is currently not supported", dir),
                ))?;
            }
            RegOperandBody::Const(_) => {}
            RegOperandBody::Sym(keyword, ..) => {
                Err(syn::Error::new(
                    keyword.span(),
                    "`sym` operand is not supported",
                ))?;
            }
            RegOperandBody::Label(keyword, ..) => {
                Err(syn::Error::new(
                    keyword.span(),
                    "`label` operand is not supported",
                ))?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AsmExpression {
    /// `format!()` call with the args already filled in for simplicity
    pub asm: TokenStream,
    pub outputs: Vec<Expression>,
    pub inputs: Vec<Expression>,
    pub options: Vec<Ident>,
}

pub fn parse_asm_call(ctx: &mut Context, tokens: TokenStream) -> syn::Result<AsmExpression> {
    let asm_spec: AsmArgs = syn::parse2(tokens)?;
    asm_spec.validate()?;

    let asm_options = asm_spec
        .operands
        .iter()
        .filter_map(|opd| match opd.operand.clone() {
            AsmOperand::AsmOptions(options) => Some(options.options.into_iter()),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();

    let registers = asm_spec.registers();

    let mut inputs = vec![];
    let mut outputs = vec![];

    for reg in registers {
        match reg.body {
            RegOperandBody::DirSpec(dir_spec, _, expr) => {
                let expr = Expression::from_expr(expr, ctx)?;
                match dir_spec {
                    DirSpec::In(_) => {
                        inputs.push(expr);
                    }
                    DirSpec::Out(_) | DirSpec::Lateout(_) => {
                        outputs.push(expr);
                    }
                }
            }
            RegOperandBody::DualDirSpec(..) => {
                unimplemented!()
            }
            RegOperandBody::Const(..) => {}
            RegOperandBody::Sym(..) | RegOperandBody::Label(..) => unimplemented!(),
        }
    }

    let mut options = vec![];

    for option in asm_options {
        options.push(format_ident!("{option}"));
    }

    let asm = asm_spec.generate_format_call()?;

    Ok(AsmExpression {
        asm,
        outputs,
        inputs,
        options,
    })
}
