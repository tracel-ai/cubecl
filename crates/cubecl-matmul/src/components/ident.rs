#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for all three tensors in a matmul
///
/// Useful to specialize some functions depending on the tensor
pub enum MatmulIdent {
    Lhs,
    Rhs,
    Out,
}

impl MatmulIdent {
    /// Equivalent to into, but type inference works better within Cube functions
    pub fn into_stage(self) -> StageIdent {
        self.into()
    }
}

// #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
// /// Identifier for all three tensors in a matmul
// ///
// /// Useful to specialize some functions depending on the tensor
// pub enum FlashIdent {
//     Query,
//     Key,
//     Value,
//     Out,
// }

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageIdent {
    Lhs,
    Rhs,
    Acc,
}

impl From<MatmulIdent> for StageIdent {
    fn from(matmul_ident: MatmulIdent) -> Self {
        match matmul_ident {
            MatmulIdent::Lhs => StageIdent::Lhs,
            MatmulIdent::Rhs => StageIdent::Rhs,
            MatmulIdent::Out => StageIdent::Acc,
        }
    }
}

// #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
// /// Identifier for all three tensors in a matmul
// ///
// /// Useful to specialize some functions depending on the tensor
// pub enum Ident {
//     Lhs,
//     Rhs,
//     Out,
// }

// impl Ident {
//     pub fn as_input_ident(&self) -> InputIdent {
//         match self {
//             Ident::Lhs => InputIdent::Lhs,
//             Ident::Rhs => InputIdent::Rhs,
//             Ident::Out => panic!("Out is not an input."),
//         }
//     }
// }

// #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
// /// Identifier for the two input tensors in a matmul.
// ///
// /// Useful to specialize some functions depending on the tensor
// pub enum InputIdent {
//     Lhs,
//     Rhs,
// }

// impl InputIdent {
//     pub fn as_ident(&self) -> Ident {
//         match self {
//             InputIdent::Lhs => Ident::Lhs,
//             InputIdent::Rhs => Ident::Rhs,
//         }
//     }
// }

// impl From<InputIdent> for Ident {
//     fn from(value: InputIdent) -> Self {
//         value.as_ident()
//     }
// }
