/// The layered matmul combines multiple component-based algorithm implementations with selection logic to pick the optimal kernel for a set of parameters.
pub mod layered;

/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;
