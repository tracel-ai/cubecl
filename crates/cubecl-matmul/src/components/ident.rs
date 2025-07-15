#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for all three tensors in a matmul
///
/// Useful to specialize some functions depending on the tensor
pub enum Ident {
    Lhs,
    Rhs,
    Out,
}

impl Ident {
    pub fn as_input_ident(&self) -> InputIdent {
        match self {
            Ident::Lhs => InputIdent::Lhs,
            Ident::Rhs => InputIdent::Rhs,
            Ident::Out => panic!("Out is not an input."),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for the two input tensors in a matmul.
///
/// Useful to specialize some functions depending on the tensor
pub enum InputIdent {
    Lhs,
    Rhs,
}

impl InputIdent {
    pub fn as_ident(&self) -> Ident {
        match self {
            InputIdent::Lhs => Ident::Lhs,
            InputIdent::Rhs => Ident::Rhs,
        }
    }
}

impl From<InputIdent> for Ident {
    fn from(value: InputIdent) -> Self {
        value.as_ident()
    }
}
