use std::marker::PhantomData;

use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, InputIdent, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;


