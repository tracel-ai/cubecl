use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Performer {}

#[derive(CubeType)]
pub struct Plane {}
#[derive(CubeType)]
pub struct Unit {}

impl Performer for Plane {}

impl Performer for Unit {}
