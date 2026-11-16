//! An attention whose scores weigh the values through relu ([`Tile::relu_weights`](cubecl_tile::Tile::relu_weights)) where its
//! holder keeps the score: each unit owning its rows in a register block, from loading to
//! writing. Each cell weighs `max(score · scale + bias, 0)` on its own, so nothing is carried
//! from one block of keys to the next and the output is never divided.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Levels, Partitioning, Semiring, SoftwareMatmul, Space, TileSpec, TileTensor, TileTensorLaunch,
    procedural::Procedural,
};

use super::online_softmax::{Attended, D, Q, S, V, inputs};
use super::{Form, implied};

#[cube(launch)]
fn unit_relu_attention(
    q: &TileTensor<'_>,
    k: &TileTensor<'_>,
    v: &TileTensor<'_>,
    out: &TileTensor<'_>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] unbiased: bool,
    #[comptime] block_keys: usize,
    #[comptime] rows: usize,
) {
    let q = q.tile(&space);
    let k = k.tile(&space).within(S, 0, keys as usize);
    let v = v.tile(&space).within(S, 0, keys as usize);
    let out = out.tile(&space);
    let bias = Procedural::new::<f32, Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in::<f32>(&space);
    let none =
        Procedural::zero::<f32>(comptime!(space.space().subspace(&[Q, S]))).tile_in::<f32>(&space);
    let value = comptime!(space.space().extent(V));
    for cube in space {
        for unit in cube {
            let (q_u, k_u, v_u, out_u, bias_u, none_u) = (
                q.at(&unit),
                k.at(&unit),
                v.at(&unit),
                out.at(&unit),
                bias.at(&unit),
                none.at(&unit),
            );
            let mut acc = out_u.register_accumulator::<f32>(
                &bias_u,
                &v_u,
                comptime!(SoftwareMatmul::new(rows * value)),
                Semiring::SUM_PROD,
            );
            acc.zero();
            for step in unit.walk().range(0, (keys as usize).div_ceil(block_keys)) {
                let (q_s, k_s, v_s, bias_s) = (
                    q_u.at(&step),
                    k_u.at(&step),
                    v_u.at(&step),
                    bias_u.at(&step),
                );
                let mut score = bias_s.register_accumulator::<f32>(
                    &q_s,
                    &k_s,
                    comptime!(SoftwareMatmul::new(rows * block_keys)),
                    Semiring::SUM_PROD,
                );
                score.zero();
                score.mma(&q_s, &k_s);
                match unbiased {
                    true => score.relu_weights::<f32>(&none_u.at(&step), scale, S),
                    false => score.relu_weights::<f32>(&bias_s, scale, S),
                }
                acc.mma(&score, &v_s);
            }
            acc.drained_into(&out_u);
        }
    }
}

/// One query row of the relu attention on the host: `Σ_j max(q · k_j · scale, 0) · v_j` over the
/// keys it sees.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    i: usize,
    (q_data, k_data, v_data): (&HostData, &HostData, &HostData),
    (queries, keys, attended): (usize, usize, usize),
    (head, value): (usize, usize),
    causal: bool,
) -> Vec<f64> {
    let scale = 1. / (head as f64).sqrt();
    let seen = |j: usize| j < attended && (!causal || j + queries <= i + attended);
    let weights: Vec<(usize, f64)> = (0..keys)
        .filter(|&j| seen(j))
        .map(|j| {
            // The keys are stored transposed.
            let dot: f64 = (0..head)
                .map(|d| q_data.get_f32(&[i, d]) as f64 * k_data.get_f32(&[d, j]) as f64)
                .sum();
            (j, (dot * scale).max(0.))
        })
        .collect();
    (0..value)
        .map(|c| {
            weights
                .iter()
                .map(|&(j, w)| w * v_data.get_f32(&[j, c]) as f64)
                .sum()
        })
        .collect()
}

struct ReluCase {
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    rows: usize,
    units: usize,
    block: usize,
    causal: bool,
    unbiased: bool,
}

fn run_units(case: ReluCase) {
    let client: Client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let ReluCase {
        queries,
        keys,
        attended,
        head,
        value,
        rows,
        units,
        block,
        causal,
        unbiased,
    } = case;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            Levels::leaf(&[(Q, rows), (S, block), (D, head), (V, value)])
                .walk_every(&[S])
                .units_distributed(Q, units)
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<f32>(&client, queries, keys, attended, head, value, true);
    let out_handle = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let scale = 1. / (head as f32).sqrt();
    let operand = |handle: cubecl::std::tensor::TensorHandle, axes| {
        TileTensorLaunch::new(
            handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(axes),
        )
    };
    unit_relu_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        operand(q_handle, &[Q, D]),
        // The keys are stored transposed, so a step's rhs runs along the keys it scores.
        operand(k_handle, &[D, S]),
        operand(v_handle, &[S, V]),
        operand(out_handle.clone(), &[Q, V]),
        attended as u32,
        queries as u32,
        scale,
        launcher.partitioning_arg(),
        causal,
        unbiased,
        block,
        rows,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for i in 0..queries {
        let expected = reference_row(
            i,
            (&q_data, &k_data, &v_data),
            (queries, keys, attended),
            (head, value),
            causal,
        );
        for (c, expected) in expected.into_iter().enumerate() {
            let got = out.get_f32(&[i, c]) as f64;
            // f32 operands summed in f32 through two contractions.
            let tolerance = 1e-4 * (1. + expected.abs());
            assert!(
                (got - expected).abs() <= tolerance,
                "row {i}, column {c}: got {got}, expected {expected}"
            );
        }
    }
}

/// **A relu attention weighs each key on its own.** Rows that walk several blocks of keys, the
/// last cut short by the attended keys and the queries masked causally, match the host's
/// `Σ relu(q · k · scale) · v`: nothing is carried across blocks, a masked key weighs zero.
#[test]
fn a_unit_relu_attention_matches_the_reference() {
    run_units(ReluCase {
        queries: 12,
        keys: 40,
        attended: 37,
        head: 96,
        value: 8,
        rows: 2,
        units: 3,
        block: 8,
        causal: true,
        unbiased: false,
    });
}

/// **An unbiased relu attention scales and cuts each cell once.** A bias zero everywhere is no
/// bias: every key is seen, each weight one multiply and one max, and a value of one column, as a
/// selector's per-head weight, sums the weights.
#[test]
fn a_unit_relu_attention_steps_unbiased_blocks() {
    run_units(ReluCase {
        queries: 8,
        keys: 32,
        attended: 32,
        head: 96,
        value: 1,
        rows: 2,
        units: 4,
        block: 8,
        causal: false,
        unbiased: true,
    });
}
