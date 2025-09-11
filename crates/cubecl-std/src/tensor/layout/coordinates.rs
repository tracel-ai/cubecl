use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

/// A set of coordinates used in layouts. Contains some utilities for comptime inspection.
#[cube]
pub trait Coordinates: CubeType + Clone {}

// Aliases for convenience and semantic clarity
pub type Coords1d = u32;
pub type Coords2d = (u32, u32);
pub type Coords3d = (u32, u32, u32);
pub type Coords4d = (u32, u32, u32, u32);
pub type Coords5d = (u32, u32, u32, u32, u32);
pub type CoordsDyn = Sequence<u32>;

impl Coordinates for Coords1d {}
impl Coordinates for Coords2d {}
impl Coordinates for Coords3d {}
impl Coordinates for Coords4d {}
impl Coordinates for Coords5d {}
impl Coordinates for CoordsDyn {}
