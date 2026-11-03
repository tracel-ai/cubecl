//! A register block as the rhs of a contraction, seeded from memory and stored without draining:
//! a state held in registers across products that read it and write it.

use super::uncut;
use crate::support::{HostData, HostDataType, TileInput};
use cubecl::{client::Client, prelude::*};
use cubecl_tile::*;

const P: Axis = Axis(0);
const K: Axis = Axis(1);
const N: Axis = Axis(2);

const BLOCK: RegisterBlock = RegisterBlock::new(256);

/// `out[p, n] = Σ_k lhs[p, k] · state[k, n]`, the state in registers.
#[cube(launch)]
fn recall_kernel(lhs: &TileArg<'_>, state: &TileArg<'_>, out: &TileArg<'_>, space: Partitioning) {
    let (lhs, state, out) = (lhs.tile(&space), state.tile(&space), out.tile(&space));
    let block = comptime!(Instruction::Registers { config: BLOCK });
    let mut held = state.accumulator::<f32>(&lhs, &state, block, Semiring::SUM_PROD);
    held.copy_from(&state);
    let mut sum = out.accumulator::<f32>(&lhs, &state, block, Semiring::SUM_PROD);
    sum.mma(&lhs, &held);
    sum.drained_into(&out);
}

/// `out[k, n] = state[k, n] + factor · (state[k, n] + lhs[0, k] · delta[0, n])`, stored twice, so
/// the first store is shown to leave the block whole: the rank-1 update, a contraction of extent
/// one, with the lhs lined along the block's rows.
#[cube(launch)]
fn update_kernel(
    lhs: &TileArg<'_>,
    delta: &TileArg<'_>,
    state: &TileArg<'_>,
    first: &TileArg<'_>,
    out: &TileArg<'_>,
    factor: f32,
    space: Partitioning,
) {
    let (lhs, delta, state) = (lhs.tile(&space), delta.tile(&space), state.tile(&space));
    let (mut first, mut out) = (first.tile(&space), out.tile(&space));
    let block = comptime!(Instruction::Registers { config: BLOCK });
    let mut held = state.accumulator::<f32>(&lhs, &delta, block, Semiring::SUM_PROD);
    held.copy_from(&state);
    let mut step = delta.accumulator::<f32>(&lhs, &delta, block, Semiring::SUM_PROD);
    step.copy_from(&delta);
    held.mma(&lhs, &step);
    first.copy_from(&held);
    held.scale_by::<f32>(factor);
    out.copy_from(&held);
}

fn arange(client: &Client, space: Space) -> TileInput {
    TileInput::builder(client, space).untiled().arange()
}

fn zeros(client: &Client, space: Space) -> TileInput {
    TileInput::builder(client, space).untiled().zeros()
}

fn cell(input: &TileInput, client: &Client, at: &[usize]) -> f32 {
    HostData::from_tensor_handle(client, input.handle(), HostDataType::F32).get_f32(at)
}

fn widths(client: &Client) -> impl Iterator<Item = usize> {
    let widest = client.properties().hardware.max_vector_size;
    [1, 2, 4].into_iter().filter(move |&w| w <= widest)
}

#[test]
fn a_block_rhs_contracts_along_an_lhs_lined_along_the_contraction() {
    let client = cubecl::test_device().client();
    let (p, k, n) = (2, 8, 8);
    let space = Space::new(&[(P, p), (K, k), (N, n)]);
    for width in widths(&client) {
        let launcher = uncut(&client, &space, &space);
        let lhs = arange(&client, space.subspace(&[P, K]));
        let state = arange(&client, space.subspace(&[K, N]));
        let out = zeros(&client, space.subspace(&[P, N]));
        recall_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            lhs.arg(&launcher, width),
            state.arg(&launcher, width),
            out.arg(&launcher, width),
            launcher.partitioning_arg(),
        );
        for row in 0..p {
            for col in 0..n {
                let want: usize = (0..k).map(|c| (row * k + c) * (c * n + col)).sum();
                assert_eq!(
                    cell(&out, &client, &[row, col]),
                    want as f32,
                    "width {width}, cell [{row}, {col}]"
                );
            }
        }
    }
}

#[test]
fn a_block_rhs_of_one_row_is_a_rank_one_update_along_the_rows() {
    let client = cubecl::test_device().client();
    let (k, n) = (8, 8);
    let factor = 0.5f32;
    let space = Space::new(&[(P, 1), (K, k), (N, n)]);
    for width in widths(&client) {
        let launcher = uncut(&client, &space, &space);
        let lhs = arange(&client, space.subspace(&[P, K]));
        let delta = arange(&client, space.subspace(&[P, N]));
        let state = arange(&client, space.subspace(&[K, N]));
        let first = zeros(&client, space.subspace(&[K, N]));
        let out = zeros(&client, space.subspace(&[K, N]));
        update_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            lhs.arg(&launcher, width),
            delta.arg(&launcher, width),
            state.arg(&launcher, width),
            first.arg(&launcher, width),
            out.arg(&launcher, width),
            factor,
            launcher.partitioning_arg(),
        );
        for row in 0..k {
            for col in 0..n {
                let want = (row * n + col + row * col) as f32;
                assert_eq!(
                    cell(&first, &client, &[row, col]),
                    want,
                    "width {width}, stored cell [{row}, {col}]"
                );
                assert_eq!(
                    cell(&out, &client, &[row, col]),
                    want * factor,
                    "width {width}, scaled cell [{row}, {col}]"
                );
            }
        }
    }
}

/// A window with fewer rows than the block leaves zeros in the rows past its end.
#[test]
fn a_seed_reads_zero_past_the_end_of_a_short_window() {
    let client = cubecl::test_device().client();
    let (real, k, n) = (3, 8, 8);
    let space = Space::new(&[(P, 1), (K, k), (N, n)]);
    let short = Space::new(&[(K, real), (N, n)]);
    for width in widths(&client) {
        let launcher = uncut(&client, &space, &space);
        let lhs = zeros(&client, space.subspace(&[P, K]));
        let delta = zeros(&client, space.subspace(&[P, N]));
        let state = arange(&client, short.clone());
        let first = zeros(&client, space.subspace(&[K, N]));
        let out = zeros(&client, space.subspace(&[K, N]));
        update_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            lhs.arg(&launcher, width),
            delta.arg(&launcher, width),
            state.arg(&launcher, width),
            first.arg(&launcher, width),
            out.arg(&launcher, width),
            1.0f32,
            launcher.partitioning_arg(),
        );
        for row in 0..k {
            for col in 0..n {
                let want = if row < real { row * n + col } else { 0 };
                assert_eq!(
                    cell(&first, &client, &[row, col]),
                    want as f32,
                    "width {width}, cell [{row}, {col}]"
                );
            }
        }
    }
}
