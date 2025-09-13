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

/// Computes both sine and cosine of an angle simultaneously.
///
/// This can be more efficient than computing sin and cos separately
/// on some GPU architectures.
///
/// # Arguments
///
/// * `val` - The angle in radians
///
/// # Returns
///
/// A tuple containing (sine, cosine) of the input angle
///
/// # Example
///
/// ```rust,ignore
/// let angle = F::new(std::f32::consts::PI / 4.0);
/// let (sin_val, cos_val) = sincos(angle);
/// ```
#[cube]
pub fn sincos<F: Float>(val: F) -> (F, F) {
    (F::sin(val), F::cos(val))
}

/// Normalizes an angle to the range [0, 2π).
///
/// # Arguments
///
/// * `angle` - The angle in radians to normalize
///
/// # Returns
///
/// The angle normalized to the range [0, 2π)
///
/// # Example
///
/// ```rust,ignore
/// let angle = F::new(3.0 * std::f32::consts::PI);
/// let normalized = normalize_angle(angle);
/// assert!((normalized - F::new(std::f32::consts::PI)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn normalize_angle<F: Float>(angle: F) -> F {
    let tau = F::new(f32::consts::TAU);
    angle - F::floor(angle / tau) * tau
}

/// Normalizes an angle to the range [-π, π).
///
/// # Arguments
///
/// * `angle` - The angle in radians to normalize
///
/// # Returns
///
/// The angle normalized to the range [-π, π)
///
/// # Example
///
/// ```rust,ignore
/// let angle = F::new(3.0 * std::f32::consts::PI);
/// let normalized = normalize_angle_signed(angle);
/// assert!((normalized - F::new(std::f32::consts::PI)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn normalize_angle_signed<F: Float>(angle: F) -> F {
    let pi = F::new(f32::consts::PI);
    let tau = F::new(f32::consts::TAU);
    let normalized = angle - F::floor(angle / tau) * tau;
    if normalized >= pi {
        normalized - tau
    } else {
        normalized
    }
}

/// Linear interpolation between two angles, taking the shortest path.
///
/// This function correctly handles the wraparound at 2π to ensure
/// the interpolation follows the shortest circular arc.
///
/// # Arguments
///
/// * `from` - The starting angle in radians
/// * `to` - The ending angle in radians
/// * `t` - The interpolation factor (0.0 = from, 1.0 = to)
///
/// # Returns
///
/// The interpolated angle
///
/// # Example
///
/// ```rust,ignore
/// let from = F::new(0.1);
/// let to = F::new(std::f32::consts::TAU - 0.1);
/// let mid = lerp_angle(from, to, F::new(0.5));
/// assert!(mid.abs() < F::new(1e-6) || (mid - F::new(std::f32::consts::TAU)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn lerp_angle<F: Float>(from: F, to: F, t: F) -> F {
    let pi = F::new(f32::consts::PI);
    let tau = F::new(f32::consts::TAU);

    let diff = to - from;
    let normalized_diff = if diff > pi {
        diff - tau
    } else if diff < -pi {
        diff + tau
    } else {
        diff
    };

    normalize_angle::<F>(from + normalized_diff * t)
}

/// Computes the shortest angular distance between two angles.
///
/// # Arguments
///
/// * `from` - The first angle in radians
/// * `to` - The second angle in radians
///
/// # Returns
///
/// The shortest angular distance, positive if `to` is clockwise from `from`
///
/// # Example
///
/// ```rust,ignore
/// let angle1 = F::new(0.1);
/// let angle2 = F::new(std::f32::consts::TAU - 0.1);
/// let distance = angle_distance(angle1, angle2);
/// assert!((distance - F::new(-0.2)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn angle_distance<F: Float>(from: F, to: F) -> F {
    let pi = F::new(f32::consts::PI);
    let tau = F::new(f32::consts::TAU);

    let diff = to - from;
    if diff > pi {
        diff - tau
    } else if diff < -pi {
        diff + tau
    } else {
        diff
    }
}

/// Smoothstep interpolation for angles.
///
/// Applies smoothstep interpolation (3t² - 2t³) between two angles,
/// taking the shortest circular path.
///
/// # Arguments
///
/// * `from` - The starting angle in radians
/// * `to` - The ending angle in radians
/// * `t` - The interpolation factor (0.0 = from, 1.0 = to)
///
/// # Returns
///
/// The smoothly interpolated angle
///
/// # Example
///
/// ```rust,ignore
/// let from = F::new(0.0);
/// let to = F::new(std::f32::consts::PI);
/// let smooth = smoothstep_angle(from, to, F::new(0.5));
/// ```
#[cube]
pub fn smoothstep_angle<F: Float>(from: F, to: F, t: F) -> F {
    let smooth_t = t * t * (F::new(3.0) - F::new(2.0) * t);
    lerp_angle::<F>(from, to, smooth_t)
}

/// Computes the angle between two 2D vectors.
///
/// # Arguments
///
/// * `x1`, `y1` - Components of the first vector
/// * `x2`, `y2` - Components of the second vector
///
/// # Returns
///
/// The angle between the vectors in radians
///
/// # Example
///
/// ```rust,ignore
/// let angle = vector_angle_2d(F::new(1.0), F::new(0.0), F::new(0.0), F::new(1.0));
/// assert!((angle - F::new(std::f32::consts::PI / 2.0)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn vector_angle_2d<F: Float>(x1: F, y1: F, x2: F, y2: F) -> F {
    let dot = x1 * x2 + y1 * y2;
    let det = x1 * y2 - y1 * x2;
    F::atan2(det, dot)
}

/// Rotates a 2D point around the origin by the given angle.
///
/// # Arguments
///
/// * `x`, `y` - The point coordinates
/// * `angle` - The rotation angle in radians
///
/// # Returns
///
/// A tuple containing the rotated coordinates (x', y')
///
/// # Example
///
/// ```rust,ignore
/// let (x, y) = rotate_2d(F::new(1.0), F::new(0.0), F::new(std::f32::consts::PI / 2.0));
/// assert!(x.abs() < F::new(1e-6) && (y - F::new(1.0)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn rotate_2d<F: Float>(x: F, y: F, angle: F) -> (F, F) {
    let cos_a = F::cos(angle);
    let sin_a = F::sin(angle);
    (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
}

/// Computes the hypotenuse of a right triangle given the lengths of the other two sides.
///
/// This function computes `sqrt(x² + y²)` in a numerically stable way that avoids
/// overflow and underflow issues.
///
/// # Arguments
///
/// * `x` - Length of one side
/// * `y` - Length of the other side
///
/// # Returns
///
/// The length of the hypotenuse
///
/// # Example
///
/// ```rust,ignore
/// let hyp = hypot(F::new(3.0), F::new(4.0));
/// assert!((hyp - F::new(5.0)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn hypot<F: Float>(x: F, y: F) -> F {
    F::sqrt(x * x + y * y)
}

/// Computes the normalized sinc function.
///
/// The sinc function is defined as:
/// - `sinc(x) = sin(πx) / (πx)` for x ≠ 0
/// - `sinc(0) = 1`
///
/// This is the normalized sinc function used in digital signal processing.
///
/// # Arguments
///
/// * `x` - The input value
///
/// # Returns
///
/// The sinc of the input
///
/// # Example
///
/// ```rust,ignore
/// let result = sinc(F::new(0.0));
/// assert!((result - F::new(1.0)).abs() < F::new(1e-6));
///
/// let result = sinc(F::new(1.0));
/// assert!(result.abs() < F::new(1e-6)); // sinc(1) ≈ 0
/// ```
#[cube]
pub fn sinc<F: Float>(x: F) -> F {
    let pi_x = F::new(f32::consts::PI) * x;
    if F::abs(x) < F::new(1e-8) {
        F::new(1.0)
    } else {
        F::sin(pi_x) / pi_x
    }
}
