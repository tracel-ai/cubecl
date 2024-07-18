/// Contains algorithms for cooperative matrix multiplication.
pub mod cmma;

/// Contains algorithms for tiling 2d matrix multiplication when cooperative matrix are not
/// available.
pub mod tiling2d;

#[cfg(feature = "export_tests")]
pub mod tests;
