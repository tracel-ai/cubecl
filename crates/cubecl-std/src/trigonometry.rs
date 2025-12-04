//! Trigonometric functions and utilities for CubeCL.
//!
//! This module provides basic trigonometric operations and angle conversion utilities
//! that can be used in all GPU kernels.

use core::f32;
use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Converts an angle from radians to degrees.
///
/// # Example
///
/// ```rust,ignore
/// let radians = F::new(std::f32::consts::PI);
/// let degrees = to_degrees(radians);
/// assert!((degrees - F::new(180.0)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn to_degrees<F: Float>(val: F) -> F {
    val * F::new(180.0 / f32::consts::PI)
}

/// Converts an angle from degrees to radians.
///
/// # Example
///
/// ```rust,ignore
/// let degrees = F::new(180.0);
/// let radians = to_radians(degrees);
/// assert!((radians - F::new(std::f32::consts::PI)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn to_radians<F: Float>(val: F) -> F {
    val * F::new(f32::consts::PI / 180.0)
}
