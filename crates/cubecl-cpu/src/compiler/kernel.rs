use std::fmt::Display;

#[derive(Debug, Default)]
pub struct MLIRKernel;

impl Display for MLIRKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "work in progress")
    }
}
