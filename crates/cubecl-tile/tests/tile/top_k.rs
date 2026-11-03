//! `Monoid::Max` and `Monoid::Min` over `TopK`, the top-k and bottom-k with their indices: the
//! `K` best in `Indexed`'s order (value, then the lower index, NaN first), padded with the
//! identity. By a unit inserting values one at a time, two halves folded apart and combined, a
//! plane's units at once and groups of four units, on two-lane vectors whose lanes reduce apart.

use cubecl::prelude::*;
use cubecl_tile::space::UnitShare;
use cubecl_tile::{Carrier, Monoid, TopK};

/// Two lanes, the best three of each.
type Best = TopK<f32, Const<2>, Const<3>>;
const K: usize = 3;

/// Item `i`'s two lanes, `input[2i]` and `input[2i + 1]`.
#[cube]
fn lanes_of(input: &[f32], i: usize) -> Vector<f32, Const<2>> {
    let mut value = Vector::<f32, Const<2>>::zeroed();
    value.insert(0usize, input[2 * i]);
    value.insert(1usize, input[2 * i + 1]);
    value
}

#[cube]
fn write(at: usize, best: Best, values: &mut [f32], indices: &mut [u32]) {
    #[unroll]
    for j in 0..3usize {
        let slot = (at * 3 + j) * 2;
        values[slot] = best.values[j].extract(0usize);
        values[slot + 1] = best.values[j].extract(1usize);
        indices[slot] = best.indices[j].extract(0usize);
        indices[slot + 1] = best.indices[j].extract(1usize);
    }
}

/// Items `start..end` inserted one at a time.
#[cube]
fn fold_range(input: &[f32], start: usize, end: usize, #[comptime] monoid: Monoid) -> Best {
    let mut best = Best::identity(monoid);
    for i in start..end {
        best = best.insert(lanes_of(input, i), Vector::new(i as u32), monoid);
    }
    best
}

/// One unit inserts every item; a second folds each half apart and combines them.
#[cube(launch)]
fn fold(input: &[f32], values: &mut [f32], indices: &mut [u32], #[comptime] monoid: Monoid) {
    let n = input.len() / 2;
    write(0, fold_range(input, 0, n, monoid), values, indices);
    let halves = monoid.combine::<Best>(
        fold_range(input, n / 2, n, monoid),
        fold_range(input, 0, n / 2, monoid),
    );
    write(1, halves, values, indices);
}

/// Each unit offers the list of its own item, and the units reduce them as `share` says.
#[cube(launch)]
fn across(
    input: &[f32],
    values: &mut [f32],
    indices: &mut [u32],
    #[comptime] share: UnitShare,
    #[comptime] monoid: Monoid,
) {
    let unit = UNIT_POS as usize;
    let own =
        Best::identity(monoid).insert(lanes_of(input, unit), Vector::new(unit as u32), monoid);
    write(unit, share.reduce::<Best>(own, monoid), values, indices);
}

/// The `K` best of `items` under `monoid`, padded with the identity.
fn expected(items: &[f32], monoid: Monoid) -> Vec<(f32, u32)> {
    let mut order: Vec<(f32, u32)> = items
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    order.sort_by(|a, b| {
        let by_value = match (a.0.is_nan(), b.0.is_nan()) {
            (true, true) => core::cmp::Ordering::Equal,
            (true, false) => core::cmp::Ordering::Less,
            (false, true) => core::cmp::Ordering::Greater,
            _ => match monoid {
                Monoid::Max => b.0.partial_cmp(&a.0).unwrap(),
                _ => a.0.partial_cmp(&b.0).unwrap(),
            },
        };
        by_value.then(a.1.cmp(&b.1))
    });
    let none = match monoid {
        Monoid::Max => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    };
    order.resize(K.max(order.len()), (none, u32::MAX));
    order.truncate(K);
    order
}

/// Interleave two lanes' items into the kernel's input.
fn interleave(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).flat_map(|(x, y)| [*x, *y]).collect()
}

/// The cases, each one lane's `n` items: lane 1 is the case reversed.
fn cases(n: usize) -> Vec<(&'static str, Vec<f32>)> {
    let base: Vec<f32> = (0..n).map(|i| (i as f32 * 7.0) % 11.0 - 5.0).collect();
    let with = |at: &[(usize, f32)]| {
        let mut v = base.clone();
        for &(i, x) in at {
            v[i] = x;
        }
        v
    };
    if n < 4 {
        // Fewer items than `K`: the list pads with the identity.
        return vec![
            ("finite", base.clone()),
            ("a NaN", with(&[(n / 2, f32::NAN)])),
            ("all -inf", vec![f32::NEG_INFINITY; n]),
        ];
    }
    vec![
        ("finite", base.clone()),
        (
            "ties",
            with(&[(1, 9.0), (n - 1, 9.0), (2, -9.0), (n - 2, -9.0)]),
        ),
        ("a NaN", with(&[(n / 2, f32::NAN)])),
        ("two NaNs", with(&[(1, f32::NAN), (n - 1, f32::NAN)])),
        ("all -inf", vec![f32::NEG_INFINITY; n]),
        ("all +inf", vec![f32::INFINITY; n]),
    ]
}

fn check(got: (f32, u32), want: (f32, u32), what: &str) {
    let same_value = (got.0.is_nan() && want.0.is_nan()) || got.0 == want.0;
    assert!(
        same_value && got.1 == want.1,
        "{what}: got {got:?}, want {want:?}"
    );
}

/// Launches over `input` with `lists` lists written, and reads them back.
fn launch(
    client: &cubecl::client::Client,
    input: &[f32],
    lists: usize,
    run: impl FnOnce(BufferArg, BufferArg, BufferArg),
) -> (Vec<f32>, Vec<u32>) {
    let slots = lists * K * 2;
    let input_handle = client.create_from_slice(f32::as_bytes(input));
    let values = client.empty(slots * 4);
    let indices = client.empty(slots * 4);
    unsafe {
        run(
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(values.clone(), slots),
            BufferArg::from_raw_parts(indices.clone(), slots),
        )
    }
    (
        f32::from_bytes(&client.read_one(values).unwrap()).to_vec(),
        u32::from_bytes(&client.read_one(indices).unwrap()).to_vec(),
    )
}

/// Inserting one item at a time, and combining two halves, both keep the `K` best.
#[test]
fn a_fold_and_a_combine_keep_the_k_best() {
    let client = cubecl::test_device().client();
    for monoid in [Monoid::Max, Monoid::Min] {
        for n in [2, 9] {
            for (name, a) in cases(n) {
                let b: Vec<f32> = a.iter().rev().copied().collect();
                let (values, indices) = launch(&client, &interleave(&a, &b), 2, |i, v, x| {
                    fold::launch(
                        &client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(1),
                        i,
                        v,
                        x,
                        monoid,
                    )
                });
                for (list, how) in ["inserted", "combined"].into_iter().enumerate() {
                    for (lane, items) in [&a, &b].into_iter().enumerate() {
                        for (j, want) in expected(items, monoid).into_iter().enumerate() {
                            let slot = (list * K + j) * 2 + lane;
                            check(
                                (values[slot], indices[slot]),
                                want,
                                &format!("{monoid:?} {name} n {n} {how} lane {lane} slot {j}"),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A plane's units, and groups of four within it, keep the `K` best of their items.
#[test]
fn a_plane_and_its_groups_keep_the_k_best() {
    let client = cubecl::test_device().client();
    let units = client.properties().hardware.plane_size_max as usize;
    if units < 4 {
        return;
    }
    for (share, group) in [
        (UnitShare::Plane, units),
        (UnitShare::Group { unit_bits: 0b11 }, 4),
    ] {
        for monoid in [Monoid::Max, Monoid::Min] {
            for (name, a) in cases(units) {
                let b: Vec<f32> = a.iter().rev().copied().collect();
                let (values, indices) = launch(&client, &interleave(&a, &b), units, |i, v, x| {
                    across::launch(
                        &client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(units as u32),
                        i,
                        v,
                        x,
                        share,
                        monoid,
                    )
                });
                for unit in 0..units {
                    let first = unit / group * group;
                    for (lane, items) in [&a, &b].into_iter().enumerate() {
                        let want = expected(&items[first..first + group], monoid);
                        for (j, (value, index)) in want.into_iter().enumerate() {
                            let index = match index {
                                u32::MAX => index,
                                index => index + first as u32,
                            };
                            let slot = (unit * K + j) * 2 + lane;
                            check(
                                (values[slot], indices[slot]),
                                (value, index),
                                &format!(
                                    "{share:?} {monoid:?} {name} unit {unit} lane {lane} slot {j}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}
