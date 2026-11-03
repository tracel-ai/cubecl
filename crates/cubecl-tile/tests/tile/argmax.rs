//! `Monoid::Max` and `Monoid::Min` over `Indexed`, the argmax and argmin: the winning value, the
//! lowest index between equal values, the first NaN when there is one. By a unit folding values
//! one at a time, a plane's units at once and groups of four units, on two-lane vectors whose
//! lanes reduce apart.

use cubecl::prelude::*;
use cubecl_tile::space::UnitShare;
use cubecl_tile::{Carrier, Indexed, Monoid};

/// Two lanes, each with its index.
type Pair = Indexed<f32, Const<2>>;

/// Item `i`, its two lanes `input[2i]` and `input[2i + 1]`, both at index `i`.
#[cube]
fn item(input: &[f32], i: usize) -> Pair {
    let mut value = Vector::<f32, Const<2>>::zeroed();
    value.insert(0usize, input[2 * i]);
    value.insert(1usize, input[2 * i + 1]);
    Pair {
        value,
        index: Vector::new(i as u32),
    }
}

#[cube]
fn write(at: usize, total: Pair, values: &mut [f32], indices: &mut [u32]) {
    values[2 * at] = total.value.extract(0usize);
    values[2 * at + 1] = total.value.extract(1usize);
    indices[2 * at] = total.index.extract(0usize);
    indices[2 * at + 1] = total.index.extract(1usize);
}

/// One unit folds every item from the identity, one at a time.
#[cube(launch)]
fn fold(input: &[f32], values: &mut [f32], indices: &mut [u32], #[comptime] monoid: Monoid) {
    let mut acc = Pair::identity(monoid);
    for i in 0..input.len() / 2 {
        acc = monoid.combine::<Pair>(acc, item(input, i));
    }
    write(0, acc, values, indices);
}

/// Each unit offers its item, and the units reduce them as `share` says; every unit writes.
#[cube(launch)]
fn across(
    input: &[f32],
    values: &mut [f32],
    indices: &mut [u32],
    #[comptime] share: UnitShare,
    #[comptime] monoid: Monoid,
) {
    let unit = UNIT_POS as usize;
    let total = share.reduce::<Pair>(item(input, unit), monoid);
    write(unit, total, values, indices);
}

/// What `Indexed` must give over `items` under `monoid`: the first NaN, else the larger (smaller)
/// value at its lowest index; `-0` and `+0` tie.
fn expected(items: &[f32], monoid: Monoid) -> (f32, u32) {
    if let Some(at) = items.iter().position(|v| v.is_nan()) {
        return (f32::NAN, at as u32);
    }
    let mut best = (
        match monoid {
            Monoid::Max => f32::NEG_INFINITY,
            _ => f32::INFINITY,
        },
        u32::MAX,
    );
    for (i, &v) in items.iter().enumerate() {
        let ahead = match monoid {
            Monoid::Max => v > best.0,
            _ => v < best.0,
        };
        if ahead || (v == best.0 && best.1 == u32::MAX) {
            best = (v, i as u32);
        }
    }
    best
}

/// Interleave two lanes' items into the kernel's input.
fn lanes(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).flat_map(|(x, y)| [*x, *y]).collect()
}

/// The cases, each one lane's `n` items: lane 1 is the case reversed, so its ties and NaNs sit
/// elsewhere.
fn cases(n: usize) -> Vec<(&'static str, Vec<f32>)> {
    let base: Vec<f32> = (0..n).map(|i| (i as f32 * 7.0) % 11.0 - 5.0).collect();
    let with = |at: &[(usize, f32)]| {
        let mut v = base.clone();
        for &(i, x) in at {
            v[i] = x;
        }
        v
    };
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
        ("zeros of both signs", with(&[(0, -0.0), (1, 0.0)])),
    ]
}

fn check(got: (f32, u32), want: (f32, u32), what: &str) {
    let same_value = (got.0.is_nan() && want.0.is_nan()) || got.0 == want.0;
    assert!(
        same_value && got.1 == want.1,
        "{what}: got {got:?}, want {want:?}"
    );
}

/// Launches `units` units over `input` and reads back each unit's two lanes.
fn launch(
    client: &cubecl::client::Client,
    input: &[f32],
    units: usize,
    run: impl FnOnce(BufferArg, BufferArg, BufferArg),
) -> (Vec<f32>, Vec<u32>) {
    let input_handle = client.create_from_slice(f32::as_bytes(input));
    let values = client.empty(units * 2 * 4);
    let indices = client.empty(units * 2 * 4);
    unsafe {
        run(
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(values.clone(), units * 2),
            BufferArg::from_raw_parts(indices.clone(), units * 2),
        )
    }
    (
        f32::from_bytes(&client.read_one(values).unwrap()).to_vec(),
        u32::from_bytes(&client.read_one(indices).unwrap()).to_vec(),
    )
}

/// Folding one item at a time gives the first winner, and an empty fold is the identity.
#[test]
fn a_unit_fold_keeps_the_lowest_index_and_the_first_nan() {
    let client = cubecl::test_device().client();
    for monoid in [Monoid::Max, Monoid::Min] {
        for (name, a) in cases(9) {
            let b: Vec<f32> = a.iter().rev().copied().collect();
            let (values, indices) = launch(&client, &lanes(&a, &b), 1, |i, v, x| {
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
            for (lane, items) in [&a, &b].into_iter().enumerate() {
                check(
                    (values[lane], indices[lane]),
                    expected(items, monoid),
                    &format!("{monoid:?} {name} lane {lane}"),
                );
            }
        }
    }
}

/// A plane's units, and groups of four within it, reduce to what the fold gives over their items.
#[test]
fn a_plane_and_its_groups_keep_the_lowest_index_and_the_first_nan() {
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
                let (values, indices) = launch(&client, &lanes(&a, &b), units, |i, v, x| {
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
                        let (value, index) = expected(&items[first..first + group], monoid);
                        let index = if index == u32::MAX {
                            index
                        } else {
                            index + first as u32
                        };
                        check(
                            (values[2 * unit + lane], indices[2 * unit + lane]),
                            (value, index),
                            &format!("{share:?} {monoid:?} {name} unit {unit} lane {lane}"),
                        );
                    }
                }
            }
        }
    }
}
