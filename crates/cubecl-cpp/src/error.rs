use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Encountered unsupported type `{0}`")]
    UnsupportedType(String),
    #[error("Encountered unsupported operation `{0}`")]
    UnsupportedOp(String),
}

pub type Result<T> = core::result::Result<T, CompileError>;

/// Errors gathered while emitting C++ source, keyed on the [`Context`](pliron::context::Context)
/// as an aux type.
///
/// `OpToCPP::to_cpp` returns a bare `String`, so an op that reaches the emitter without a
/// lowering has no way to report upward. It used to `.unwrap()`, which panics on whichever
/// thread is compiling — a thread the caller never joins. The launch then reports success, the
/// kernel never runs, and its output buffer keeps whatever it already held, so the failure
/// surfaces as silently wrong numbers rather than a compile error. Recording here instead lets
/// `compile_ir` fail the compilation properly once emission finishes.
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
