// use crate::components::global::dummy::MaskReader;
// use crate::components::stage::StageAttentionConfig;
// use crate::components::tile::TileAttention;
// use cubecl_core as cubecl;
// use cubecl_core::prelude::*;
// use cubecl_std::tensor::layout::{Coordinates, Coords2d};
// use cubecl_std::{CubeOption, CubeOptionExpand};
// use std::marker::PhantomData;

// use crate::components::global::GlobalAttentionConfig;
// use crate::components::{AttentionPrecision, AttentionTilingScheme};

// #[derive(CubeType)]
// pub enum AttentionMask {
//     /// Full mask tensor in global memory.
//     /// Used when the user provides an explicit mask.
//     /// Causal or out-of-bounds padding are applied directly in the materialized mask
//     //
//     // Materialized(MaskReader<AP, G>, LogicalMask),
//     Materialized(LogicalMask),

//     /// Mask is applied logically.
//     /// This variant is chosen when no mask tensor is provided but the attention logic
//     /// requires masking for causal or padding purposes.
//     Logical(LogicalMask),

//     /// No mask is applied at all.
//     /// Used when neither a mask tensor is provided nor causal/padding masking is needed.
//     None,
// }

// #[cube]
// impl AttentionMask {
//     pub fn new(
//         #[comptime] causal: bool,
//         out_of_bounds: CubeOption<Coords2d>,
//         #[comptime] tiling_scheme: AttentionTilingScheme,
//     ) -> AttentionMask {
//         // TODO materialized case
//         if comptime!(causal || out_of_bounds.is_some()) {
//             AttentionMask::new_Logical(LogicalMask::new(causal, out_of_bounds, tiling_scheme))
//         } else {
//             AttentionMask::new_None()
//         }
//     }

//     pub fn to_stage(&self, row: u32, col: u32) -> AttentionMask {
//         match self {
//             AttentionMask::Materialized(logical_mask) => {
//                 // Adjust origin to the view?
//                 // Advance mask reader's iterator
//                 todo!()
//             }
//             AttentionMask::Logical(logical_mask) => {
//                 AttentionMask::new_Logical(logical_mask.to_stage(row, col))
//             }
//             AttentionMask::None => AttentionMask::new_None(),
//         }
//     }

//     pub fn to_partition(&self, row: u32) -> AttentionMask {
//         match self {
//             AttentionMask::Materialized(logical_mask) => {
//                 // Adjust origin
//                 todo!()
//             }
//             AttentionMask::Logical(logical_mask) => {
//                 AttentionMask::new_Logical(logical_mask.to_partition(row))
//             }
//             AttentionMask::None => AttentionMask::new_None(),
//         }
//     }

//     pub fn to_tile(&self, row: u32, col: u32) -> AttentionMask {
//         match self {
//             AttentionMask::Materialized(logical_mask) => {
//                 // Load tile from global memory to register
//                 // Using view, iterator, origin and row,col
//                 todo!()
//             }
//             AttentionMask::Logical(logical_mask) => {
//                 AttentionMask::new_Logical(logical_mask.to_tile(row, col))
//             }
//             AttentionMask::None => AttentionMask::new_None(),
//         }
//     }

//     pub fn apply<E: Numeric>(&self, pos_in_tile: Coords2d) -> E {
//         let should_mask = match self {
//             AttentionMask::Materialized(logical_mask) => {
//                 // registers[pos_in_tile]
//                 todo!()
//             }
//             AttentionMask::Logical(logical_mask) => logical_mask.should_mask(pos_in_tile),
//             // TODO refactor so it does not do the addition of +0
//             AttentionMask::None => false,
//         };

//         E::cast_from(should_mask) * E::min_value()
//     }
// }

// #[derive(CubeType, Copy, Clone)]
// pub struct LogicalMask {
//     origin: Coords2d,
//     #[cube(comptime)]
//     pub causal: bool,
//     pub out_of_bounds: CubeOption<Coords2d>,
//     #[cube(comptime)]
//     tiling_scheme: AttentionTilingScheme,
// }

// #[cube]
// impl LogicalMask {
//     pub fn new(
//         #[comptime] causal: bool,
//         out_of_bounds: CubeOption<Coords2d>,
//         #[comptime] tiling_scheme: AttentionTilingScheme,
//     ) -> LogicalMask {
//         LogicalMask {
//             origin: (0u32, 0u32).runtime(),
//             causal,
//             out_of_bounds,
//             tiling_scheme,
//         }
//     }

//     pub fn to_stage(&self, row: u32, col: u32) -> LogicalMask {
//         let q_factor = comptime!(self.tiling_scheme.elements_in_stage_seq_q());
//         let kv_factor = comptime!(self.tiling_scheme.elements_in_stage_seq_kv());

//         LogicalMask {
//             origin: Coords2d::add(self.origin, (row * q_factor, col * kv_factor)),
//             causal: self.causal,
//             out_of_bounds: self.out_of_bounds,
//             tiling_scheme: self.tiling_scheme,
//         }
//     }

//     pub fn to_partition(&self, row: u32) -> LogicalMask {
//         let q_factor = comptime!(self.tiling_scheme.elements_in_partition_seq_q());

//         LogicalMask {
//             origin: Coords2d::add(self.origin, (row * q_factor, 0u32)),
//             causal: self.causal,
//             out_of_bounds: self.out_of_bounds,
//             tiling_scheme: self.tiling_scheme,
//         }
//     }

//     pub fn to_tile(&self, row: u32, col: u32) -> LogicalMask {
//         let q_factor = comptime!(self.tiling_scheme.elements_in_tile_seq_q());
//         let kv_factor = comptime!(self.tiling_scheme.elements_in_tile_seq_kv());

//         LogicalMask {
//             origin: Coords2d::add(self.origin, (row * q_factor, col * kv_factor)),
//             causal: self.causal,
//             out_of_bounds: self.out_of_bounds,
//             tiling_scheme: self.tiling_scheme,
//         }
//     }

//     pub fn should_mask(&self, pos_in_tile: Coords2d) -> bool {
//         let pos = Coords2d::add(self.origin, pos_in_tile);

//         let causal_masked = self.causal && pos.0 < pos.1;

//         let oob_masked = match self.out_of_bounds {
//             CubeOption::Some(bounds) => !Coords2d::is_in_bounds(&pos, &bounds),
//             CubeOption::None => false,
//         };

//         causal_masked || oob_masked
//     }
// }
