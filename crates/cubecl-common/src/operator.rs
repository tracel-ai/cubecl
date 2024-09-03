use derive_more::derive::Display;

/// An operator used in the intermediate representaion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum Operator {
    // Arithmetic
    /// Add (+) operator
    Add,
    /// Sub (-) operator
    Sub,
    /// Mul (*) operator
    Mul,
    /// Div (/) operator
    Div,
    /// Rem (%) operator
    Rem,

    // Arithmetic Assign
    /// Add assign (+=) operator
    AddAssign,
    /// Sub assign (-=) operator
    SubAssign,
    /// Mul assing (*=) operator
    MulAssign,
    /// Div assign (/=) operator
    DivAssign,
    /// Rem assign (%=) operator
    RemAssign,

    // Comparison
    /// Equals (==) operator
    Eq,
    /// Not equal (!=) operator
    Ne,
    /// Less than (<) operator
    Lt,
    /// Less than equals (<=) operator
    Le,
    /// Greater than equal (>=) operator
    Ge,
    /// Greater than (>) operator
    Gt,

    // Boolean
    /// And (&&) operator
    And,
    /// Or (||) operator
    Or,
    /// Bitwise XOR (^) operator
    BitXor,
    /// Bitwise And (&) operator
    BitAnd,
    /// Bitwise Or (|) operator
    BitOr,

    // Boolean assign
    /// Bitwise xor assign (^=) operator
    BitXorAssign,
    /// Bitwise and assign (&=) operator
    BitAndAssign,
    /// Bitwise or assign (|=) operator
    BitOrAssign,

    /// Shift left (<<) operator
    Shl,
    /// Shift right (>>) operator
    Shr,
    /// Shift left assign (<<=) operator
    ShlAssign,
    /// Shift right assign (>>= operator)
    ShrAssign,

    // Unary
    /// Dereference operator (*)
    Deref,
    /// Not operator (!)
    Not,
    /// Negation unary operator (-)
    Neg,

    // Function-like
    /// The cosign operator
    Cos,
    /// Min operator
    Min,
    /// Max operator
    Max,
}

impl Operator {
    /// Whether this is an assign op, aka whether the output is the same as the left hand side
    pub fn is_assign(&self) -> bool {
        matches!(
            self,
            Operator::AddAssign
                | Operator::SubAssign
                | Operator::MulAssign
                | Operator::DivAssign
                | Operator::RemAssign
                | Operator::BitXorAssign
                | Operator::BitAndAssign
                | Operator::BitOrAssign
                | Operator::ShlAssign
                | Operator::ShrAssign
                | Operator::Deref
                | Operator::Not
                | Operator::Neg
        )
    }
}
