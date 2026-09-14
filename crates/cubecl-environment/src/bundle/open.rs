use alloc::boxed::Box;
use std::path::Path;

use crate::bytes::Bytes;

use super::{Bundle, BundleError, EmbeddedBundle};

/// Opens a bundle file for [`import`](super::import).
///
/// Applications that ship a bundle can open [`EmbeddedBundle`] directly; this
/// is for the cases that take a path from a user.
///
/// The returned bundle is what [`import`](super::import) consumes, which is
/// the only thing a bundle is for.
///
/// ```no_run
/// # fn main() -> Result<(), cubecl_environment::bundle::BundleError> {
/// let bundle = cubecl_environment::bundle::open("h100.bundle")?;
/// let report = cubecl_environment::bundle::import(bundle.as_ref());
/// # Ok(())
/// # }
/// ```
pub fn open<P: AsRef<Path>>(path: P) -> Result<Box<dyn Bundle>, BundleError> {
    let bytes = Bytes::from_bytes_vec(std::fs::read(path)?);

    Ok(Box::new(EmbeddedBundle::open(bytes)?))
}
