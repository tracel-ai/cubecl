use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Encountered unsupported type `{0}`")]
    UnsupportedType(String),
    #[error("Encountered unsupported operation `{0}`")]
    UnsupportedOp(String),
}

pub type Result<T> = core::result::Result<T, CompileError>;
