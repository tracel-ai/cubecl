use alloc::boxed::Box;
use std::path::Path;

use crate::bytes::Bytes;

use super::{Bundle, BundleError, EmbeddedBundle, SqliteBundle, flat_bundle_version};

/// Opens a bundle file, in whichever format it was written.
///
/// Applications that ship a bundle usually know its format and can open
/// [`SqliteBundle`] or [`EmbeddedBundle`] directly. This is for the cases that
/// take a path from a user, where both formats are equally likely.
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
    let path = path.as_ref();

    if !is_flat(path)? {
        return Ok(Box::new(SqliteBundle::open(path)?));
    }

    let bytes = Bytes::from_bytes_vec(std::fs::read(path)?);

    Ok(Box::new(EmbeddedBundle::open(bytes)?))
}

/// Whether `path` carries the flat bundle header.
///
/// The format is decided from the header before either reader runs, so a
/// malformed bundle surfaces the error of the reader that matches it instead
/// of every failure reading as "not a cubecl flat bundle".
fn is_flat(path: &Path) -> Result<bool, BundleError> {
    use std::io::Read;

    let mut header = [0u8; 12];
    // `read_exact`, not `read`: a single `read` may return a short buffer even
    // for a regular file, which would misclassify a valid flat bundle as
    // SQLite. A file too small to hold the header can't be a flat bundle.
    match std::fs::File::open(path)?.read_exact(&mut header) {
        Ok(()) => Ok(flat_bundle_version(&header).is_some()),
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(err) => Err(err.into()),
    }
}
