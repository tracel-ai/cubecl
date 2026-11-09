//! The colours of the `CubeCL` page on the Tracel website, token for token: the
//! one palette the command line's reports, its help and the terminal browser
//! each paint with, so the three and the page read as one product. Only the
//! colours that read on any background are here: the page's own background
//! and body text are the terminal's, left as the user set them.

use clap::builder::styling::{Color, RgbColor};

/// A colour of the page, by its red, green and blue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rgb(pub u8, pub u8, pub u8);

/// `--color-text-secondary`: labels, headers, hints.
pub(crate) const MUTED: Rgb = Rgb(0x89, 0x8d, 0x95);
/// `--color-text-dark`, the hue of `--color-border-soft`: the rules.
pub(crate) const RULE: Rgb = Rgb(0x4d, 0x53, 0x60);
/// `--color-product-cubecl`: the accent.
pub(crate) const ACCENT: Rgb = Rgb(0xaa, 0x79, 0xb8);

impl From<Rgb> for Color {
    fn from(Rgb(red, green, blue): Rgb) -> Self {
        RgbColor(red, green, blue).into()
    }
}
