use crate::support::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, TileInput, ValidationResult,
    assert_equals_approx,
};
use crate::tile::{Form, implied, uncut};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::{bytes::Bytes, ir::ElemType, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_tile::{
    Axis, Levels, Partitioning, Projection, Space, StageStorage, Tile, TileArg, TileArgLaunch,
    TileSpec, kind::Field, layout::PhysicalAxisMap,
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// `M` and `N` as blocks make them: which block, and where inside it.
const MB: Axis = Axis(2);
const MI: Axis = Axis(3);
const NB: Axis = Axis(4);
const NI: Axis = Axis(5);

/// Base sanity: a plain (non-quantized) tile copies through unchanged.
#[test]
fn copy_non_quantized_matches_reference() {
    let (m, n) = (8, 8);
    let client = cubecl::test_device().client();
    let space = Space::new(&[(M, m), (N, n)]);

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    let launch = uncut(&client, &space, &space);
    plain_copy::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        input.arg(&launch, 1),
        output.arg(&launch, 1),
        launch.partitioning_arg(),
    );

    let input_host = HostData::from_tensor_handle(&client, input.handle(), HostDataType::F32);
    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    assert_equals_approx(&got, &input_host, 1e-6).enforce();
}

/// The op walks the levels that hand regions to different cubes and stops at the plane level,
/// which the transport spreads over the cube itself. Stepping the plane level would leave every
/// plane but the first unwritten, since the fill indexes by the flat unit.
#[test]
fn copy_spread_across_cubes_and_planes_matches_reference() {
    let (m, n) = (4, 512);
    let client = cubecl::test_device().client();
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n)]),
            Levels::leaf(&[(N, 32), (M, 1)])
                .planes(&[(N, 4)])
                .cubes(&[N, M])
                .build(),
        ),
        Form::Static,
    );
    let space = launch.space().clone();

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    plain_copy::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        input.arg(&launch, 1),
        output.arg(&launch, 1),
        launch.partitioning_arg(),
    );

    let input_host = HostData::from_tensor_handle(&client, input.handle(), HostDataType::F32);
    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    assert_equals_approx(&got, &input_host, 1e-6).enforce();
}

#[cube(launch)]
/// A plain (non-quantized) copy: both tiles serve their element straight from their tensors.
pub fn plain_copy(input: &TileArg<'_>, output: &TileArg<'_>, space: Partitioning) {
    let input = input.tile(&space);
    let mut output = output.tile(&space);
    output.copy_from(&input);
}

/// `out = values ⊗ scales`, stated where the kernel copies them.
#[cube(launch)]
fn block_scaled_copy(
    values: &TileArg<'_>,
    scales: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
) {
    let values = values.tile(&space);
    let scales = scales.tile(&space);
    let mut output = output.tile(&space);
    output.dequantize_from(&values.mul(&scales));
}

/// `out = values ⊗ scales ⊗ global`: two levels, innermost first, bound as one operand whose
/// values carry their block scales, which carry the global one.
#[cube(launch)]
fn two_level_scaled_copy(values: &TileArg<'_>, output: &TileArg<'_>, space: Partitioning) {
    let values = values.tile(&space);
    let mut output = output.tile(&space);
    output.dequantize_from(&values);
}

/// Block-scaled packed values: each `bm×bn` block carries its own scale, looked up at the
/// value's own coordinate once `M` and `N` are split into blocks, `out == q · scale[i/bm, j/bn]`,
/// and with `global` set, `· global` on top.
fn check_block_scaled_copy(m: usize, n: usize, bm: usize, bn: usize, global: Option<f32>) {
    let client = cubecl::test_device().client();
    let scheme = QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
    let pack = scheme.num_quants();
    let max = client.properties().hardware.max_vector_size;
    if pack > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the packing factor ({pack})"
        ))
        .enforce();
        return;
    }

    let values = TileInput::builder(&client, Space::new(&[(M, m), (N, n)]))
        .untiled()
        .packed(&scheme)
        .arange();
    let output = TestInput::builder(client.clone(), Shape::from(vec![m, n]))
        .zeros()
        .generate_without_host_data();
    let space = Space::new(&[(MB, m / bm), (MI, bm), (NB, n / bn), (NI, bn)]);
    let projection = Projection::new(
        &[MB, MI, NB, NI],
        &[
            PhysicalAxisMap::disjoint(&[(MB, bm), (MI, 1)]),
            PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
        ],
    );
    let scales = projection.scales_per(MB).scales_per(NB);
    let launch = uncut(&client, &space, &space);
    let dtype = f32::elem_type_native();
    let output_spec = TileSpec::new(projection.clone());
    match global {
        None => block_scaled_copy::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            values.values_arg(1, TileSpec::new(projection.clone())),
            values.scales_arg(1, TileSpec::new(scales)),
            TileArgLaunch::new(
                output.clone().binding().into_tensor_arg(),
                dtype,
                pack,
                output_spec,
            ),
            launch.partitioning_arg(),
        ),
        Some(g) => {
            let global = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
                .custom(vec![g])
                .generate_without_host_data();
            let per_tensor = Projection::new(&[MB, MI, NB, NI], &[PhysicalAxisMap::broadcast()]);
            two_level_scaled_copy::launch(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                values
                    .values_arg(1, TileSpec::new(projection.clone()))
                    .scales(values.scales_arg(1, TileSpec::new(scales)))
                    .scales(TileArgLaunch::new(
                        global.binding().into_tensor_arg(),
                        dtype,
                        1,
                        TileSpec::new(per_tensor),
                    )),
                TileArgLaunch::new(
                    output.clone().binding().into_tensor_arg(),
                    dtype,
                    pack,
                    output_spec,
                ),
                launch.partitioning_arg(),
            )
        }
    }

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    let (sn, g) = (n / bn, global.unwrap_or(1.0));
    let shape = Shape::from(vec![m, n]);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    values.q[k] as f32 * values.scale_values[(i / bm) * sn + (j / bn)] * g
                })
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };
    assert_equals_approx(&got, &expected, 1e-6).enforce();
}

/// One scale per block, over grids of blocks along either axis or both.
#[test]
fn a_copy_of_block_scaled_values_decodes_them() {
    check_block_scaled_copy(8, 8, 4, 4, None); // a 2×2 grid of blocks
    check_block_scaled_copy(8, 8, 8, 4, None); // blocks along N only
    check_block_scaled_copy(8, 8, 4, 8, None); // blocks along M only
    check_block_scaled_copy(16, 8, 4, 4, None); // a 4×2 grid
}

/// Block scales under a per-tensor one: the expectation carries the global scale, so a copy
/// that dropped it fails by exactly that factor.
#[test]
fn a_copy_of_two_level_scaled_values_decodes_them() {
    check_block_scaled_copy(8, 8, 4, 4, Some(0.5));
    check_block_scaled_copy(16, 8, 4, 4, Some(0.25));
    check_block_scaled_copy(4, 4, 4, 4, Some(0.5)); // the window is one block
}

/// A zero per-tensor scale zeroes every value, so the global level provably takes part in each
/// one rather than defaulting to one.
#[test]
fn a_zero_global_scale_zeroes_the_copy() {
    check_block_scaled_copy(8, 8, 4, 4, Some(0.0));
}

/// The batch a cube stands in, one wide in each cube's region.
const B: Axis = Axis(6);

/// `stage = values ⊗ scales` over a cube's region, then the stage out: the decode stated where the
/// stage is filled, as a kernel staging a packed cache block by block does.
#[cube(launch)]
fn batched_scaled_stage<O: Numeric>(
    values: &TileArg<'_>,
    scales: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
    #[define(O)] _dtype: ElemType,
) {
    let decoded = values.tile(&space).mul(&scales.tile(&space));
    let output = output.tile(&space);
    for cube in &space {
        let mut stage = Tile::smem::<O>(
            comptime!(Space::new(&[(M, rows), (N, cols)])),
            output.vector_size(),
            StageStorage::Strided,
            0usize,
        );
        stage.dequantize_from(&decoded.at(&cube));
        sync_cube();
        let mut out = output.at(&cube);
        out.copy_from(&stage);
    }
}

/// A cube's region of a batched operand spans the batch one wide, and its stage spans only the
/// matrix: the decoding copy drops the one-wide axis rather than refusing the two boxes as
/// different, even with the batch dynamic.
#[test]
fn a_scaled_region_decodes_into_a_stage_without_its_one_wide_batch() {
    let (batches, rows, cols) = (3, 4, 16);
    let client = cubecl::test_device().client();
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, batches), (M, rows), (N, cols)]),
            Levels::leaf(&[(B, 1), (M, rows), (N, cols)])
                .cubes(&[B])
                .build(),
        ),
        Form::DynamicAlong(&[B]),
    );

    let count = batches * rows * cols;
    let values: Vec<i8> = (0..count).map(|i| ((i * 7) % 255) as i8).collect();
    let words: Vec<u32> = values
        .chunks(4)
        .map(|w| {
            w.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u8 as u32) << (j * 8)))
        })
        .collect();
    let mut binding = TensorHandle::new_contiguous(
        vec![batches, rows, cols / 4],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    )
    .binding();
    // Declared in values: the buffer holds words, the operand addresses what they pack.
    binding.shape = vec![batches, rows, cols].into();
    binding.strides = vec![rows * cols, cols, 1].into();
    let scales: Vec<f32> = (0..batches * rows)
        .map(|i| (i as f32 + 1.0) / 8.0)
        .collect();
    let (scales_t, _) = TestInput::builder(client.clone(), Shape::from(vec![batches, rows]))
        .dtype(f32::elem_type_native())
        .custom(scales.clone())
        .generate_with_f32_host_data();
    let output = TestInput::builder(client.clone(), Shape::from(vec![batches, rows, cols]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    batched_scaled_stage::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        TileArgLaunch::new(
            binding.into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[B, M, N]).packed(Field::Quant(QuantValue::Q8S)),
        ),
        TileArgLaunch::new(
            scales_t.binding().into_tensor_arg(),
            f32::elem_type_native(),
            1,
            TileSpec::direct(&[B, M]),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            f32::elem_type_native(),
            4,
            TileSpec::direct(&[B, M, N]),
        ),
        launch.partitioning_arg(),
        rows,
        cols,
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for b in 0..batches {
        for m in 0..rows {
            for n in 0..cols {
                let want = values[(b * rows + m) * cols + n] as f32 * scales[b * rows + m];
                let have = got.get_f32(&[b, m, n]);
                assert!(
                    (have - want).abs() < 1e-5,
                    "at ({b}, {m}, {n}): got {have}, want {want}"
                );
            }
        }
    }
}
