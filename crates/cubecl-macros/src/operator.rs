use std::fmt::Display;

/// An operator used in the intermediate representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Mul assign (*=) operator
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
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl Operator {
    /// Whether this is an assign op, aka whether the output is the same as the
    /// left hand side
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
        )
    }

    /// Get the expanded op name for this operation
    pub fn op_name(&self) -> String {
        if self.is_assign() {
            let name = self.to_string().to_lowercase();
            format!("{}_assign_op", &name[..name.len() - 6])
        } else {
            self.to_string().to_lowercase()
        }
    }

    /// Get the expanded op name for this array operation
    pub fn array_op_name(&self) -> String {
        if self.is_assign() {
            let name = self.to_string().to_lowercase();
            format!("{}_assign_array_op", &name[..name.len() - 6])
        } else {
            self.to_string().to_lowercase()
        }
    }
}
