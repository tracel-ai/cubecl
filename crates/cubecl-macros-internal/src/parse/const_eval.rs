use syn::{
    ExprClosure, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token,
};

pub struct ConstEval {
    pub op_type: Type,
    _comma: Token![,],
    _brace: token::Brace,
    pub arms: Punctuated<ConstEvalArm, Token![,]>,
}

pub struct ConstEvalArm {
    pub attr_types: TypeList,
    _colon: Token![:],
    pub closure: ExprClosure,
}

pub struct TypeList(pub Punctuated<AttrType, Token![,]>);

pub struct AttrType {
    pub attr_ty: Type,
    pub subtypes: Punctuated<Type, Token![,]>,
}

impl Parse for ConstEval {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        Ok(ConstEval {
            op_type: input.parse()?,
            _comma: input.parse()?,
            _brace: syn::braced!(content in input),
            arms: content.parse_terminated(ConstEvalArm::parse, Token![,])?,
        })
    }
}

impl Parse for ConstEvalArm {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ConstEvalArm {
            attr_types: input.parse()?,
            _colon: input.parse()?,
            closure: input.parse()?,
        })
    }
}

impl Parse for TypeList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let types = if input.peek(token::Bracket) {
            let content;
            syn::bracketed!(content in input);
            content.parse_terminated(AttrType::parse, Token![,])?
        } else {
            let mut types = Punctuated::new();
            types.push(input.parse()?);
            types
        };
        Ok(Self(types))
    }
}

impl Parse for AttrType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attr_ty: Type = input.parse()?;
        let subtypes = if input.peek(token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            content.parse_terminated(Type::parse, Token![,])?
        } else {
            Punctuated::new()
        };
        Ok(Self { attr_ty, subtypes })
    }
}
