pub const SKEW: Skew = Skew::None;

#[derive(Clone, Copy)]
/// Strategy for adding padding to shared memory in order to avoid bank conflicts.
///
/// On GPUs, shared memory is organized into multiple memory banks (typically 32),
/// and when multiple threads in a warp access addresses that map to the same bank,
/// access becomes serialized (a **bank conflict**), which degrades performance.
///
/// This enum describes how to pad shared memory tiles to minimize such conflicts.
pub enum Skew {
    /// Adds `X` extra lines (line_size of the stage, typically 1) at the end of each segment, i.e.
    /// row (if row-major layout) or column (if column-major layout) of a shared memory tile.
    ///
    /// This disrupts regular memory alignment that could lead to threads in the same warp
    /// accessing memory addresses in the same bank, thus **reducing or eliminating bank conflicts**.
    ///
    /// For example:
    /// - If the row stride is 16 and 32 threads are accessing rows in parallel,
    ///   adding 1 extra element (i.e., `Pad(1)`) will make the stride 17, which avoids
    ///   conflicts from strides divisible by 32.
    ///
    /// Note: Use values like `1` for typical conflict-avoidance padding,
    /// especially when row or column length is a multiple of 32.
    Pad(u32),

    /// No padding is added.
    ///
    /// Use this to minimize shared memory size.
    None,
}

impl Skew {
    pub fn padding_size(&self) -> u32 {
        match self {
            Skew::Pad(x) => *x,
            Skew::None => 0,
        }
    }
}
