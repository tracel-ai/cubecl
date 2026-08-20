use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Encountered unsupported type `{0}`")]
    UnsupportedType(String),
    #[error("Encountered unsupported operation `{0}`")]
    UnsupportedOp(String),
}

pub type Result<T> = core::result::Result<T, CompileError>;

/// Errors gathered while emitting C++ source, stored on the [`Context`](pliron::context::Context)
/// as an aux type. Emission runs under `Display` and can't fail, so ops without a lowering are
/// recorded here and `compile_ir` fails the compilation once emission finishes.
#[derive(Default)]
pub struct EmissionErrors(core::cell::RefCell<Vec<CompileError>>);

impl EmissionErrors {
    /// Record an error hit while emitting an operation.
    pub fn record(&self, error: CompileError) {
        self.0.borrow_mut().push(error);
    }

    /// Take everything recorded so far, leaving the list empty.
    pub fn take(&self) -> Vec<CompileError> {
        core::mem::take(&mut self.0.borrow_mut())
    }
}
