//! `Monoid::Max` and `Monoid::Min` on floats: NaN kept, `∓inf` identities, infinities and signed
//! zeros left as they are, by a unit folding values one at a time and by a plane's units at once,
//! a whole plane and one only partly launched.

use cubecl::prelude::*;
use cubecl_tile::Monoid;
use cubecl_tile::space::UnitShare;

/// One unit folds `input` from the monoid's identity, combining one value at a time.
#[allow(clippy::needless_range_loop)] // a kernel loop indexes; it cannot iterate a slice
#[cube(launch)]
fn fold(input: &[f32], out: &mut [f32], #[comptime] monoid: Monoid) {
    let mut acc = Monoid::identity::<f32>(monoid);
    for i in 0..input.len() {
        acc = monoid.combine::<f32>(acc, input[i]);
    }
    out[0] = acc;
}

/// Each unit offers one value of `input`, and the plane reduces them.
#[cube(launch)]
fn plane_reduce(input: &[f32], out: &mut [f32], #[comptime] monoid: Monoid) {
    let unit = UNIT_POS as usize;
    let total = UnitShare::Plane.reduce::<f32>(input[unit], monoid);
    if unit == 0 {
        out[0] = total;
    }
}

fn values(units: usize, with: &[(usize, f32)]) -> Vec<f32> {
    let mut values: Vec<f32> = (0..units).map(|u| (u as f32 * 7.0) % 11.0 - 5.0).collect();
    for &(at, value) in with {
        values[at] = value;
    }
    values
}

/// What the monoid must give for `input`: NaN if any is, else the plain max or min.
fn expected(input: &[f32], monoid: Monoid) -> f32 {
    if input.iter().any(|v| v.is_nan()) {
        return f32::NAN;
    }
    let pick = |a: f32, b: f32| match monoid {
        Monoid::Max => {
            if b > a {
                b
            } else {
                a
            }
        }
        _ => {
            if b < a {
                b
            } else {
                a
            }
        }
    };
    let identity = match monoid {
        Monoid::Max => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    };
    input.iter().copied().fold(identity, pick)
}

fn same(got: f32, want: f32) -> bool {
    (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
}

fn run(client: &cubecl::client::Client, input: &[f32], monoid: Monoid, plane: bool) -> f32 {
    let handle = client.create_from_slice(f32::as_bytes(input));
    let out = client.empty(4);
    let (input_arg, out_arg) = unsafe {
        (
            BufferArg::from_raw_parts(handle, input.len()),
            BufferArg::from_raw_parts(out.clone(), 1),
        )
    };
    match plane {
        true => plane_reduce::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(input.len() as u32),
            input_arg,
            out_arg,
            monoid,
        ),
        false => fold::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            input_arg,
            out_arg,
            monoid,
        ),
    }
    f32::from_bytes(&client.read_one(out).unwrap())[0]
}

/// The cases every path must agree on, `units` values each.
fn cases(units: usize) -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("finite", values(units, &[])),
        ("a NaN", values(units, &[(units / 2 + 1, f32::NAN)])),
        ("a NaN first", values(units, &[(0, f32::NAN)])),
        ("+inf", values(units, &[(1, f32::INFINITY)])),
        ("-inf", values(units, &[(1, f32::NEG_INFINITY)])),
        ("all -inf", vec![f32::NEG_INFINITY; units]),
        ("all +inf", vec![f32::INFINITY; units]),
        ("all -0", vec![-0.0; units]),
        (
            "+inf and a NaN",
            values(units, &[(0, f32::INFINITY), (units - 1, f32::NAN)]),
        ),
    ]
}

/// Folding one value at a time keeps NaN, and an empty fold is the identity, `∓inf`.
#[test]
fn a_unit_fold_keeps_nan_and_starts_at_infinity() {
    let client = cubecl::test_device().client();
    for monoid in [Monoid::Max, Monoid::Min] {
        for (name, input) in cases(9) {
            let got = run(&client, &input, monoid, false);
            let want = expected(&input, monoid);
            assert!(same(got, want), "{monoid:?} {name}: got {got}, want {want}");
        }
    }
}

/// A plane's units reduce to the same, whether the plane is launched whole or only partly: the
/// plane instruction drops NaN, so it is put back, and nothing else changes.
#[test]
fn a_plane_keeps_nan_whole_or_partly_launched() {
    let client = cubecl::test_device().client();
    let plane = client.properties().hardware.plane_size_max as usize;
    if plane < 2 {
        return;
    }
    for units in [plane, plane / 2 + 1] {
        for monoid in [Monoid::Max, Monoid::Min] {
            for (name, input) in cases(units) {
                let got = run(&client, &input, monoid, true);
                let want = expected(&input, monoid);
                assert!(
                    same(got, want),
                    "{monoid:?} {name} over {units} units: got {got}, want {want}"
                );
            }
        }
    }
}
