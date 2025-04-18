pub const SKEW: Skew = Skew::Line(1);
// pub const SKEW: Skew = Skew::Element(1);
// pub const SKEW: Skew = Skew::None;

#[derive(Clone, Copy)]
/// Strategy for adding padding to shared memory in order to avoid bank conflicts.
pub enum Skew {
    /// Adds `X` extra elements to each segment
    ///
    /// Note: in the current implementation, this will fix the stage line size to 1
    Element(u32),

    /// Adds `X` extra lines to each segment
    ///
    /// Note: in the current implementation, this will fix the stage line size to the global line size
    Line(u32),

    /// No padding is added.
    None,
}

impl Skew {
    pub fn padding_size(&self) -> u32 {
        // This may look wrong, but in the current implementation
        // if it's `Element` then the line size is 1, whereas if it's
        // `Line` then the line size is kept.
        // In both cases it corresponds to 1 line
        match self {
            Skew::Element(x) => *x,
            Skew::Line(x) => *x,
            Skew::None => 0,
        }
    }
}
