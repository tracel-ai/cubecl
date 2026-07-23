use syn::{
    ExprClosure, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token,
};

pub struct Simplify {
    pub op_type: Type,
    _comma: Token![,],
    _brace: token::Brace,
    pub arms: Punctuated<ExprClosure, Token![,]>,
}

impl Parse for Simplify {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Simplify {
            op_type: input.parse()?,
            _comma: input.parse()?,
            _brace: syn::braced!(content in input),
            arms: content.parse_terminated(ExprClosure::parse, Token![,])?,
        })
    }
}
