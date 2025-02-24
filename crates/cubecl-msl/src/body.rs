use super::Instruction;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct Body {
    pub instructions: Vec<Instruction>,
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for ops in self.instructions.iter() {
            write!(f, "{ops}")?;
        }

        Ok(())
    }
}
