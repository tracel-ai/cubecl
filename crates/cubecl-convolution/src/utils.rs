/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}
