//! Variable-length sequences as a plain windowed walk.
//!
//! Sequences of different lengths packed end to end on one axis. The kernel walks that axis with
//! [`Walk::window`], whose base and length are runtime values, so a sequence steps its own tokens
//! and nothing else: no side channel on an operand, no displaced window origin, `at` unchanged.
//!
//! The reduce is the shape that isolates it: the output omits the packed axis, so nothing but the
//! walk's bounds decides which tokens a sequence reduces.
#![allow(non_snake_case)]

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::Shape};
use cubecl_tile::*;

/// Which sequence.
const B: Axis = Axis(0);
/// The packed position: every sequence's tokens, end to end in one buffer.
const P: Axis = Axis(1);
/// The feature a token carries.
const D: Axis = Axis(2);

/// `out[b] = Σ packed[p]` over the tokens of sequence `b`, whose packed range the cumulative
/// lengths give as `[ends[b], ends[b + 1])`.
#[cube(launch)]
fn ragged_sum_kernel<E: Numeric>(
    packed: &TileArg<'_>,
    out: &TileArg<'_>,
    ends: &Tensor<u32>,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] token: Level,
    #[define(E)] _dtype: ElemType,
) {
    let packed = packed.tile(&space);
    let out = out.tile(&space);

    for sequence in space.over(&seq) {
        let b = sequence.coord(B);
        let start = ends[b] as usize;
        let end = ends[b + 1] as usize;

        let mut total = out.at(&sequence);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        // The whole of it: the walk over the packed axis takes `end - start` steps from `start`,
        // so an empty sequence takes none and a short one never reads its neighbour's tokens.
        let tokens = sequence.over(&token).range(start, end - start);
        for pos in tokens {
            let mut cell = out.at(&pos);
            cell.reduce_axis_accumulate(&packed.at(&pos), comptime!(Monoid::Sum));
        }
    }
}

/// The four sequences every case here packs, as cumulative lengths: 2, 0, 5 and 3 tokens.
/// The zero-length one is the edge that a walk gets right by taking no steps at all.
const ENDS: [u32; 5] = [0, 2, 2, 7, 10];
const TOKENS: usize = 10;
const SEQS: usize = 4;
const FEATURES: usize = 4;

/// Every packed cell distinct, so reducing one token of a neighbouring sequence cannot land on the
/// right answer by accident.
fn packed_values() -> Vec<f32> {
    (0..TOKENS * FEATURES)
        .map(|i| {
            let (p, d) = (i / FEATURES, i % FEATURES);
            (p * 10 + d) as f32
        })
        .collect()
}

fn run() -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, SEQS), (P, TOKENS), (D, FEATURES)]),
            Levels::leaf(&[(B, 1), (P, 1)])
                .walk_every(&[P])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (packed_handle, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32_ty)
        .custom(packed_values())
        .generate_with_f32_host_data();

    let ends_handle = TestInput::builder(client.clone(), Shape::new([ENDS.len()]))
        .dtype(u32_ty)
        .custom(ENDS.iter().map(|&e| e as f32).collect())
        .generate_without_host_data();

    // Poisoned, not zeroed: the kernel owes every cell a value, including the empty sequence's.
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQS * FEATURES])
        .generate_without_host_data();

    ragged_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            packed_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[P, D]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, D]),
        ),
        ends_handle.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference, reduced on the host from the same cumulative lengths.
fn expected(ends: &[u32]) -> Vec<Vec<f32>> {
    let values = packed_values();
    (0..SEQS)
        .map(|b| {
            (0..FEATURES)
                .map(|d| {
                    (ends[b] as usize..ends[b + 1] as usize)
                        .map(|p| values[p * FEATURES + d])
                        .sum()
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_windowed_walk_reduces_each_sequence_over_its_own_tokens() {
    let got = run();
    let want = expected(&ENDS);
    for (b, row) in want.iter().enumerate() {
        for (d, cell) in row.iter().enumerate() {
            assert_eq!(
                got.get_f32(&[b, d]),
                *cell,
                "sequence {b} feature {d}: tokens {}..{}",
                ENDS[b],
                ENDS[b + 1]
            );
        }
    }
}

/// Tokens in blocks of four, which is where a step count alone stops being enough: sequence 2
/// starts at token 2 and sequence 3 at token 7, neither a block boundary, and both run a length
/// no block edge lands on.
const BLOCK: usize = 4;

/// `out[b] = Σ packed[p]` again, but a block of tokens per step. [`Tile::within`] places the
/// operand at the sequence's first token and stops its reads at the last, so the block that
/// overruns reads zero instead of the next sequence.
#[cube(launch)]
fn blocked_ragged_sum_kernel<E: Numeric>(
    packed: &TileArg<'_>,
    out: &TileArg<'_>,
    ends: &Tensor<u32>,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] token: Level,
    #[define(E)] _dtype: ElemType,
) {
    let packed = packed.tile(&space);
    let out = out.tile(&space);

    for sequence in space.over(&seq) {
        let b = sequence.coord(B);
        let start = ends[b] as usize;
        let end = ends[b + 1] as usize;

        let mut total = out.at(&sequence);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        let seq_view = packed.within(P, start, end);

        for blk in sequence
            .over(&token)
            .range(0, (end - start).div_ceil(BLOCK))
        {
            let mut cell = out.at(&blk);
            cell.reduce_axis_accumulate(&seq_view.at(&blk), comptime!(Monoid::Sum));
        }
    }
}

fn run_blocked(ends: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, SEQS), (P, TOKENS), (D, FEATURES)]),
            Levels::leaf(&[(B, 1), (P, BLOCK)])
                .walk_every(&[P])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (packed_handle, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32_ty)
        .custom(packed_values())
        .generate_with_f32_host_data();
    let ends_handle = TestInput::builder(client.clone(), Shape::new([ends.len()]))
        .dtype(u32_ty)
        .custom(ends.iter().map(|&e| e as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQS * FEATURES])
        .generate_without_host_data();

    blocked_ragged_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            packed_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[P, D]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, D]),
        ),
        ends_handle.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// Two sets of lengths over the same buffer, both with sequences starting off a block boundary
/// and running past one. The answer follows the lengths, which it could not if the placement did
/// nothing, since every sequence would reduce from token zero under both.
#[test]
fn a_placed_window_reduces_each_blocked_sequence_over_its_own_tokens() {
    for ends in [ENDS, [0, 3, 3, 6, 10]] {
        let got = run_blocked(&ends);
        let want = expected(&ends);
        for (b, row) in want.iter().enumerate() {
            for (d, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, d]),
                    *cell,
                    "sequence {b} feature {d}: tokens {}..{}",
                    ends[b],
                    ends[b + 1]
                );
            }
        }
    }
}
