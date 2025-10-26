use core::fmt;

/// Errors that can occur when trying to launch QR decomposition.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum QRSetupError {
    /// The input should be a matrix where m should be greater or equal to n.
    InvalidShape,
}

impl fmt::Display for QRSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape => write!(
                f,
                "The input should be a matrix where m should be greater or equal to n."
            ),
        }
    }
}
