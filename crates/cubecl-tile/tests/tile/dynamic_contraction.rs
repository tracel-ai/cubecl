//! A register block contracting a weight lined along the contraction, checked against the host
//! with the contracted axis's extent read at compile time and at launch.
//!
//! `out = x · w`, the weight stored `[n, k]` and read as `[k, n]`, each step four lines of `K`.
//! Two walks of it: every unit taking a column's whole `K`, and a column's units taking its steps
//! in turns, their partials summed by the drain, as `matmul_mb` does. A `k` whose last step
//! reaches past the end is not a case: an operand lined along an axis it overhangs is refused.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const VECTOR: usize = 4;
const STEP: usize = VECTOR * 4;
const PLANES: usize = 4;

const FORMS: [Form<'static>; 4] = [
    Form::Static,
    Form::DynamicAlong(&[N]),
    Form::DynamicAlong(&[K]),
    Form::DynamicAlong(&[N, K]),
];

/// One register block over the whole nest, summed at every leaf and drained once. The two levels
/// under the planes are the units and the walk, in whichever order the walk puts them.
#[cube(launch)]
fn contract<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] budget: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = x.tile(&space);
    let b = w.tile(&space);
    let c = out.tile(&space);
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers {
            config: RegisterBlock::new(budget)
        }),
        Semiring::SUM_PROD,
    );
    for cube in space {
        for plane in cube {
            for outer in plane {
                for leaf in outer {
                    let mut acc_leaf = acc.at(&leaf);
                    acc_leaf.mma(&a.at(&leaf), &b.at(&leaf));
                }
            }
        }
    }
    acc.drained_into(&c);
}

#[derive(Clone, Copy, Debug)]
enum Walk {
    WholeKPerUnit,
    KSplitAcrossUnits,
}

fn x_value(r: usize, i: usize) -> f32 {
    ((r * 5 + i) % 7) as f32 - 3.0
}

fn w_value(j: usize, i: usize) -> f32 {
    ((j * 3 + i) % 5) as f32 - 2.0
}

/// `x · w` computed on the host, `m × n` in row order.
fn host_product([m, k, n]: [usize; 3]) -> impl Iterator<Item = f32> {
    (0..m * n).map(move |idx| {
        let (r, j) = (idx / n, idx % n);
        (0..k).map(|i| x_value(r, i) * w_value(j, i)).sum::<f32>()
    })
}

/// `x · w` in `dtype` under `form` on a `plane`-unit plane, read back as `m × n`.
fn run(
    walk: Walk,
    plane: usize,
    [m, k, n]: [usize; 3],
    form: Form<'_>,
    dtype: ElemType,
) -> Vec<f32> {
    let client = cubecl::test_device().client();
    let input = |shape, values: Vec<f32>| {
        TestInput::builder(client.clone(), shape)
            .dtype(dtype)
            .custom(values)
            .generate_with_f32_host_data()
            .0
    };
    let x = input(
        shape![m, k],
        (0..m * k).map(|idx| x_value(idx / k, idx % k)).collect(),
    );
    let w = input(
        shape![n, k],
        (0..n * k).map(|idx| w_value(idx / k, idx % k)).collect(),
    );
    // A value no product reaches, so a cell no unit wrote shows.
    let out = input(shape![m, n], vec![-1e6; m * n]);

    // A one-unit plane cuts nothing, and a level states only what it cuts; the split walk is
    // then the whole-`K` one.
    let leaf = Levels::leaf(&[(M, m), (N, 1), (K, STEP)]);
    let levels = match (walk, plane) {
        (_, 1) => leaf.walk_every(&[K]).units(&[]),
        (Walk::WholeKPerUnit, _) => leaf.walk_every(&[K]).units(&[(N, plane)]),
        (Walk::KSplitAcrossUnits, _) => leaf.units(&[(K, plane)]).interleaved(K).walk_every(&[K]),
    };
    let space = Space::new(&[(M, m), (N, n), (K, k)]);
    let partitioning = Partitioning::new(space, levels.planes(&[(N, PLANES)]).cubes(&[N]).build());
    let launcher = implied(&client, partitioning, form);

    let a = launcher
        .arg(x.binding(), dtype)
        .axes(&[M, K])
        .vectorize(VECTOR)
        .build()
        .unwrap();
    let mut weight = w.binding();
    weight.shape = [k, n].into();
    weight.strides = [1, k].into();
    let b = launcher
        .arg(weight, dtype)
        .axes(&[K, N])
        .in_stride_order()
        .vectorize(VECTOR)
        .build()
        .unwrap();
    let c = launcher
        .arg(out.clone().binding(), dtype)
        .axes(&[M, N])
        .vectorize(1)
        .build()
        .unwrap();

    contract::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        m * VECTOR,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    (0..m * n)
        .map(|idx| got.get_f32(&[idx / n, idx % n]))
        .collect()
}

/// Every form at one row and eight, at a `k` of two chunks of the split walk and one of three,
/// which is no power of two, on an `n` the whole-`K` walk's cubes overhang and one they tile.
/// Reports every case that differs from the host rather than stopping at the first.
fn check(walk: Walk) {
    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if max < VECTOR {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {VECTOR}-wide weight lines"
        ))
        .enforce();
        return;
    }
    let plane = client.properties().hardware.plane_size_max as usize;
    let chunk = STEP * plane;
    let ns: &[usize] = match walk {
        Walk::WholeKPerUnit => &[2352, 2048],
        Walk::KSplitAcrossUnits => &[2048],
    };
    let mut failures = Vec::new();
    for form in FORMS {
        for m in [1, 8] {
            for k in [2 * chunk, 3 * chunk] {
                for &n in ns {
                    let wrong = run(walk, plane, [m, k, n], form, f32::elem_type_native())
                        .iter()
                        .zip(host_product([m, k, n]))
                        .filter(|(g, w)| (*g - w).abs() > 1e-3)
                        .count();
                    if wrong > 0 {
                        failures.push(format!("{form:?} m={m} k={k} n={n}: {wrong} cells"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{walk:?}: differs from the host at {failures:#?}"
    );
}

#[test]
fn whole_k_per_unit_in_every_form() {
    check(Walk::WholeKPerUnit);
}

#[test]
fn k_split_across_units_in_every_form() {
    check(Walk::KSplitAcrossUnits);
}

/// An `i32` block sums exactly in `i32` under either walk. An `fma` is a float instruction: over
/// integers it compiles to nothing on some backends and every cell reads zero.
#[test]
fn an_i32_block_sums_exactly() {
    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if max < VECTOR {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {VECTOR}-wide weight lines"
        ))
        .enforce();
        return;
    }
    let plane = client.properties().hardware.plane_size_max as usize;
    let shape = [8, 2 * STEP * plane, 2048];
    let want: Vec<f32> = host_product(shape).collect();
    for walk in [Walk::WholeKPerUnit, Walk::KSplitAcrossUnits] {
        let got = run(walk, plane, shape, Form::Static, i32::elem_type_native());
        assert_eq!(got, want, "{walk:?}: an i32 block differs from the host");
    }
}
