//! An axis spanning several dims of a plain buffer: `[a, k, c]` read as `B = a · c` rows of `k`,
//! `B` spanning the dims `a` and `c` on either side of `k`, its coordinate split across them. The
//! axes run in the order of their innermost dims, `[K, B]`, so lines run along `c`.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::shape};

use super::{Form, implied};
use cubecl_tile::launch::Refusal;
use cubecl_tile::*;

const B: Axis = Axis(0);
const K: Axis = Axis(1);

/// Every region of `input` copied into `out`, one value a line.
#[cube(launch)]
fn copy_values_kernel(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let x = input.tile(&space);
    let o = out.tile(&space);
    for region in space.over(&level) {
        let window = x.at(&region);
        let lines = window.flat::<f32, Const<1>>();
        let mut written = o.at(&region);
        let mut cells = written.flat_mut::<f32, Const<1>>();
        for i in 0..lines.shape() {
            cells.write(i, lines.read(i));
        }
    }
}

/// Every region of `input` copied into `out`, four values a line.
#[cube(launch)]
fn copy_lines_kernel(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let x = input.tile(&space);
    let o = out.tile(&space);
    for region in space.over(&level) {
        let window = x.at(&region);
        let lines = window.flat::<f32, Const<4>>();
        let mut written = o.at(&region);
        let mut cells = written.flat_mut::<f32, Const<4>>();
        for i in 0..lines.shape() {
            cells.write(i, lines.read(i));
        }
    }
}

/// `[a, k, c]` copied into a buffer of the same shape, both bound with `B` spanning `a` and `c`,
/// in boxes of `rows` rows read `width` values a line, over the space with `b_last` deciding
/// whether `B` is its last axis: the copy is exact.
fn folded_copy(a: usize, k: usize, c: usize, rows: usize, width: usize, b_last: bool) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let data: Vec<f32> = (0..a * k * c).map(|i| i as f32).collect();
    let (input, _) = TestInput::builder(client.clone(), shape![a, k, c])
        .dtype(f32_ty)
        .custom(data.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![a, k, c])
        .dtype(f32_ty)
        .custom(vec![-1.0; a * k * c])
        .generate_without_host_data();
    let space = match b_last {
        true => Space::new(&[(K, k), (B, a * c)]),
        false => Space::new(&[(B, a * c), (K, k)]),
    };
    let launcher = implied(
        &client,
        Partitioning::new(
            space,
            Levels::leaf(&[(K, 1), (B, rows)])
                .walk_every(&[K, B])
                .build(),
        ),
        Form::Static,
    );
    let bound = |binding: TensorBinding| {
        launcher
            .arg(binding, f32_ty)
            .axes(&[B, K, B])
            .vectorize(width)
            .build()
            .expect("an axis spanning two dims is served along its innermost one")
    };
    let input = bound(input.binding());
    let output = bound(out.clone().binding());
    match width {
        1 => copy_values_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            input.arg(),
            output.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        ),
        _ => copy_lines_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            input.arg(),
            output.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        ),
    }
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for i in 0..a {
        for j in 0..k {
            for l in 0..c {
                let want = data[i * k * c + j * c + l];
                assert_eq!(got.get_f32(&[i, j, l]), want, "at [{i}, {j}, {l}]");
            }
        }
    }
}

#[test]
fn a_row_at_a_time_reads_in_place() {
    folded_copy(3, 5, 8, 1, 1, false);
}

#[test]
fn a_box_of_rows_inside_the_inner_dim_reads_in_place() {
    folded_copy(3, 5, 8, 4, 1, false);
}

#[test]
fn lines_of_rows_inside_the_inner_dim_read_in_place() {
    folded_copy(3, 5, 8, 4, 4, false);
}

#[test]
fn lines_of_rows_read_in_place_whatever_the_space_order() {
    folded_copy(3, 5, 8, 4, 4, true);
}

/// An axis spanning several dims stated out of the order of its innermost dims is refused:
/// its coordinates and its line would disagree.
#[test]
fn axes_out_of_dim_order_are_refused() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let input = TestInput::builder(client.clone(), shape![3, 5, 8])
        .dtype(f32_ty)
        .custom(vec![0.0; 3 * 5 * 8])
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, 24), (K, 5)]),
            Levels::leaf(&[(K, 1), (B, 4)]).walk_every(&[K, B]).build(),
        ),
        Form::Static,
    );
    let refused = launcher
        .arg(input.binding(), f32_ty)
        .gathered(Projection::tiled(&[B, K], &[B, K, B]))
        .vectorize(4)
        .build()
        .err();
    assert!(
        matches!(refused, Some(Refusal::AxesOutOfDimOrder { .. })),
        "{refused:?}"
    );
}
