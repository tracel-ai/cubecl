use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use super::super::prologue::RuntimeCmmaInfo;
use crate::matmul::cmma::config::ComptimeCmmaInfo;

#[cube]
pub(crate) trait LoadInfo {
    fn smem_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn smem_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn tile_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn tile_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn smem_tile_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn smem_tile_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn num_tile_elements(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32;
    fn batch_offset(runtime_info: RuntimeCmmaInfo) -> u32;
    fn dim_vertical(runtime_info: RuntimeCmmaInfo) -> u32;
    fn dim_horizontal(runtime_info: RuntimeCmmaInfo) -> u32;
    fn skips(k_offset: u32, runtime_info: RuntimeCmmaInfo) -> (u32, u32);
}

pub(crate) struct LhsLoadInfo {}
pub(crate) struct RhsLoadInfo {}

#[cube]
impl LoadInfo for LhsLoadInfo {
    fn smem_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.block_size_m
    }

    fn smem_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.block_size_k
    }

    fn tile_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.tile_size_m
    }

    fn tile_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.tile_size_k
    }

    fn smem_tile_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        Self::smem_height(comptime_info) / Self::tile_height(comptime_info)
    }

    fn smem_tile_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        Self::smem_width(comptime_info) / Self::tile_width(comptime_info)
    }

    fn num_tile_elements(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        Self::tile_height(comptime_info) * Self::tile_width(comptime_info)
    }

    fn batch_offset(runtime_info: RuntimeCmmaInfo) -> u32 {
        runtime_info.offsets.batch_lhs
    }

    fn dim_vertical(runtime_info: RuntimeCmmaInfo) -> u32 {
        runtime_info.dims.m
    }

    fn dim_horizontal(runtime_info: RuntimeCmmaInfo) -> u32 {
        runtime_info.dims.k
    }

    fn skips(k_offset: u32, runtime_info: RuntimeCmmaInfo) -> (u32, u32) {
        (runtime_info.offsets.cube_row, k_offset)
    }
}

#[cube]
impl LoadInfo for RhsLoadInfo {
    fn smem_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.block_size_k
    }

    fn smem_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.block_size_n
    }

    fn tile_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.tile_size_k
    }

    fn tile_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        comptime_info.tile_size_n
    }

    fn smem_tile_height(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        Self::smem_height(comptime_info) / Self::tile_height(comptime_info)
    }

    fn smem_tile_width(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        Self::smem_width(comptime_info) / Self::tile_width(comptime_info)
    }

    fn num_tile_elements(#[comptime] comptime_info: ComptimeCmmaInfo) -> u32 {
        Self::tile_height(comptime_info) * Self::tile_width(comptime_info)
    }

    fn batch_offset(runtime_info: RuntimeCmmaInfo) -> u32 {
        runtime_info.offsets.batch_rhs
    }

    fn dim_vertical(runtime_info: RuntimeCmmaInfo) -> u32 {
        runtime_info.dims.k
    }

    fn dim_horizontal(runtime_info: RuntimeCmmaInfo) -> u32 {
        runtime_info.dims.n
    }

    fn skips(k_offset: u32, runtime_info: RuntimeCmmaInfo) -> (u32, u32) {
        (k_offset, runtime_info.offsets.cube_col)
    }
}
