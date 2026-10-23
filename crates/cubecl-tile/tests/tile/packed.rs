//! `TileSpec::packed`: an operand whose values are fields of a stored word, said on its own.
//!
//! A packed tensor is *values*, stored small. Saying so takes one fact (how wide a field is and
//! how it reads back) that belongs to the values, not to a quantization scheme: there are no
//! scales here. The tile serves the values the words hold ([`TileArg::tile`]) and the read unpacks.
//!
//! What a *quantized* operand adds on top is its scales, which are their own tensor and their own
//! operand; folding them in is a verb the kernel writes ([`Tile::mm_scaled`], see
//! [`scaled`](super::scaled)). The matmuls here are spelled that way, up to a whole decode gemv.
//!
//! Every test packs its own words: values low field first, `32 / bits` of them per word, and a
//! binding whose shape counts *values* while the packing says how many sit in one stored word.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    bytes::Bytes, features::TypeUsage, ir::FloatKind, ir::types::Fp8Format, prelude::*,
    quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
};
use cubecl_common::{e2m1, e4m3, e5m2};

use crate::tile::uncut;
use cubecl_tile::Instruction;
use cubecl_tile::kind::Field;
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;
use half::f16;

use super::matmul::require_cmma_8x8x8_f32;
use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contraction as the two axes a scale block makes of it: which block, and where inside it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);
/// The columns, likewise, where a scale covers a block of them.
const NB: Axis = Axis(4);
const NI: Axis = Axis(5);

/// The register block every contraction here runs under.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// A packed operand copied into a plain one: the words unpack at the read, and nothing in the
/// kernel, the spec or the launch mentions a scale.
#[cube(launch)]
fn packed_copy(input: &TileArg<'_>, output: &TileArg<'_>, space: Partitioning) {
    let input = input.tile(&space);
    let mut output = output.tile(&space);
    output.copy_from(&input);
}

/// `c = (w ⊗ s) · x` with `w` packed: the q4 kernel in this spelling. Three tensors, three
/// operands, one verb.
#[cube(launch)]
fn packed_matmul(
    w: &TileArg<'_>,
    x: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let w = w.tile(&space).mul(&scale.tile(&space));
    let x = x.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_r.mma(&w.at(&region), &x.at(&region));
    }
}

/// [`packed_matmul`] with two scale levels: `nvfp4`'s shape.
#[cube(launch)]
fn nvfp4_shaped_matmul(
    w: &TileArg<'_>,
    x: &TileArg<'_>,
    blocks: &TileArg<'_>,
    global: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    // Two levels, said twice: the blocks, then the factor over the whole tensor.
    let w = w
        .tile(&space)
        .mul(&blocks.tile(&space))
        .mul(&global.tile(&space));
    let x = x.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_r.mma(&w.at(&region), &x.at(&region));
    }
}

/// **`nvfp4`'s shape, end to end.** `e2m1` values eight to a word, a scale per sixteen of them, and
/// one factor over the whole tensor.
///
/// The only thing standing between this and the format itself is where the block scales are
/// *stored*: `nvfp4` puts them in `ue4m3`, which needs a device that loads it. Everything else is
/// here: the decode, two ordered levels, the coarser spanning no axis, 16-blocks on 8-value words.
///
/// Nothing in the kernel names the format, the block, or the number of levels.
///
/// Every operand is bound one element wide, so unlike the read tests below this asks nothing of
/// the device's vector cap and runs everywhere. That matters: the decode it covers lowers
/// differently per backend, so a device that skipped this is the one most worth running it on.
#[test]
fn nvfp4_shaped_decode() {
    let (field, rows, cols, block, blocks) = (QuantValue::E2M1, 4, 4, 16, 2);
    let depth = block * blocks;
    let factor = 32 / field.size_bits();

    let client = cubecl::test_device().client();

    // Every code, cycled, so the whole value ladder and both signs are read back.
    let codes: Vec<u32> = (0..rows * depth).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &c)| acc | (c << (j * field.size_bits())))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves and a quarter, so the reference is exact.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let g = vec![0.25f32];

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let (g_tensor, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(g.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    nvfp4_shaped_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(row, block of sixteen)`.
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            g_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // The same axes, addressing neither: one value over all their tiles.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    let value = e2m1::from_bits(codes[m * depth + k] as u8).to_f32();
                    value * s[m * blocks + k / block] * g[0] * x[k * cols + n]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// `c = x · (w ⊗ s)` with `w` packed along its columns: the q4 kernel with the *weights on the
/// right*, which is the shape the shipped quant matmul has. Same verb, same body: the scales are
/// written on the other factor, and their axes are checked against it.
#[cube(launch)]
fn packed_matmul_rhs(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let x = x.tile(&space);
    let w = w.tile(&space).mul(&scale.tile(&space));
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_r.mma(&x.at(&region), &w.at(&region));
    }
}

/// `c = (w ⊗ s) · x` with `w` an `i8` tensor: the native store, which needs no packing statement
/// at all. The binding says `i8`, the tile serves `i8`, and the contraction casts each value into
/// the accumulator's element as always: a value is whatever its tensor holds, as a scale is.
#[cube(launch)]
fn native_matmul(
    w: &TileArg<'_>,
    x: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let w = w.tile(&space);
    let x = x.tile(&space);
    let w = w.mul(&scale.tile(&space));
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_r.mma(&w.at(&region), &x.at(&region));
    }
}

/// The decode gemv, whole: one row of activations against packed weights read straight from
/// global memory, their scales beside them, accumulating in a register block opened above the
/// walk. `N` spreads across cubes, which is all a gemv has to spread.
#[cube(launch)]
fn packed_gemv<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let values = w.tile(&space);
    let w = values.mul(&scale.tile(&space));
    let c = c.tile(&space);
    for cube in space {
        let x = x.at(&cube);
        let values = values.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        // The accumulator lives in registers across the whole walk and drains once.
        let acc = c.accumulator::<E>(
            &x,
            &values,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma(&x.at(&step), &w.at(&step));
        }
        acc.drained_into(&c);
    }
}

/// [`packed_matmul`] with the scales stored as bytes, four to a word, served one at a time: a
/// `ue8m0` or `ue4m3` scale is read in its own width, and nothing widens it on the host.
#[cube(launch)]
fn packed_matmul_byte_scales(
    w: &TileArg<'_>,
    x: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let w = w.tile(&space).mul(&scale.tile(&space));
    let x = x.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_r.mma(&w.at(&region), &x.at(&region));
    }
}

/// [`packed_gemv`] with byte scales on the rhs, into the promoted block.
#[cube(launch)]
fn packed_gemv_byte_scales<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let values = w.tile(&space);
    let w = values.mul(&scale.tile(&space));
    let c = c.tile(&space);
    for cube in space {
        let x = x.at(&cube);
        let values = values.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        let acc = c.accumulator::<E>(
            &x,
            &values,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma(&x.at(&step), &w.at(&step));
        }
        acc.drained_into(&c);
    }
}

/// The prefill shape on the tensor cores: a packed weight on the rhs with byte scales, landed
/// unpacked and scaled by the plane's units, contracted through the plain cmma instruction.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn packed_cmma_rhs<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    // Both factors land on their own: a fragment reads a window as it lies, and a gmem layout is
    // unchecked.
    let x = x.tile(&space);
    let w = w.tile(&space).mul(&scale.tile(&space));
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(&x, &w, Instruction::Cmma, Semiring::SUM_PROD);
    // The level cuts the columns into two fragments and walks `K`: unrolled, so each region
    // selects its fragment at comptime.
    for region in space.over(&level).unrolled() {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&x.at(&region), &w.at(&region));
    }
    acc.drained_into(&c);
}

/// [`packed_cmma_rhs`] with the weight dequantized at `element` by the kernel, before it
/// contracts.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn packed_cmma_rhs_dequantized<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] element: ElemType,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let w = w.tile(&space).mul(&scale.tile(&space));
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(&x, &w, Instruction::Cmma, Semiring::SUM_PROD);
    for region in space.over(&level).unrolled() {
        let mut acc_r = acc.at(&region);
        let w_r = acc_r.dequantized(&w.at(&region), element);
        acc_r.mma(&x.at(&region), &w_r);
    }
    acc.drained_into(&c);
}

/// Four 8-bit values per word.
#[test]
fn eight_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q8S, 8, 8);
    let bits = field.size_bits();
    // A packed line serves a whole word, so this is the width the tile is read at.
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    // Cycling through every value a signed `bits`-wide field represents, sign extension included.
    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    // Shape and strides count values; the packing says how many share a stored word.
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(field),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        uncut(&client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = values[m * cols + n] as f32;
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// Eight 4-bit values per word: the q4 store, and the one the plan names.
#[test]
fn four_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q4S, 8, 32);
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(field),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        uncut(&client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = values[m * cols + n] as f32;
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// Eight `e2m1` codes per word, reinterpreted rather than sign-extended.
///
/// The field of `nvfp4` and `mxfp4`. Its bits are an index into the format's sixteen values, not a
/// small integer, so reading it as one would answer plausible nonsense (`0b0111` is `6.0`, and as
/// a signed nibble `7`). Nothing states a scheme: the packing names the field, which decodes.
#[test]
fn fp4_codes_unpack_on_read() {
    let client = cubecl::test_device().client();
    check_fp4_codes_unpack(&client, Field::from(QuantValue::E2M1));
}

/// The same codes as an operand bound at launch reads them on this device ([`Field::read_on`]):
/// through its own `e2m1x2` conversion where it has one, placed from their bits where it would
/// emulate one.
#[test]
fn fp4_codes_unpack_as_the_device_reads_them() {
    let client = cubecl::test_device().client();
    let field = Field::from(QuantValue::E2M1).read_on(&client);
    if field == Field::from(QuantValue::E2M1) {
        ValidationResult::Skipped(
            "the device reads e2m1 as the plain field, which fp4_codes_unpack_on_read checks"
                .to_string(),
        )
        .enforce();
        return;
    }
    check_fp4_codes_unpack(&client, field);
}

/// The same codes placed from their bits as `f16` and lifted, on every device: the decode a
/// device that emulates `e2m1x2` reads, held to every code whichever device runs the suite.
#[test]
fn fp4_codes_unpack_placed() {
    let client = cubecl::test_device().client();
    check_fp4_codes_unpack(
        &client,
        Field::Placed {
            value: QuantValue::E2M1,
            lifted: true,
        },
    );
}

/// Every `e2m1` code, packed eight to a word, unpacked through `field` and checked against the
/// format's own decode.
fn check_fp4_codes_unpack(client: &Client, field: Field) {
    let (rows, cols) = (4, 32);
    let bits = field.size_bits();
    let factor = 32 / bits;

    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    // Every code, cycled, so both signs and the whole value ladder are read back.
    let codes: Vec<u32> = (0..rows * cols).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &c)| acc | (c << (j * bits)))
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(field),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        uncut(client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = e2m1::from_bits(codes[m * cols + n] as u8).to_f32();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// Sixteen 2-bit values per word.
#[test]
fn two_bit_fields_unpack_on_read() {
    let (field, rows, cols) = (QuantValue::Q2S, 4, 32);
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );

    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(field),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        uncut(&client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = values[m * cols + n] as f32;
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The q4 matmul.** A packed weight tensor, its scales beside it as their own tensor, and one
/// contraction: `C[m,n] = Σ_k W[m,k] · S[m, k/block] · X[k,n]`.
#[test]
fn a_packed_operand_contracts_against_its_scales() {
    let (field, rows, cols, block, blocks) = (QuantValue::Q4S, 4, 4, 8, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, so an f16 scale would be exact too.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // A region sits inside one block, and the packed line is one word of it.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    packed_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(row, block)`: `KI` is carried and addresses nothing.
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The same contraction over 8-bit fields, so the packing factor is not what makes it work.
#[test]
fn eight_bit_fields_contract_against_their_scales() {
    let (field, rows, cols, block, blocks) = (QuantValue::Q8S, 4, 4, 8, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, factor)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    packed_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The folded walk takes its scales several at a time.** A packed line folds a whole word per
/// step, so nothing walks the contraction one value at a time. Here the scales are read two to a
/// line, each field covering a block the walk picks.
#[test]
fn a_folded_walk_takes_its_scales_several_at_a_time() {
    let (field, rows, cols, block, blocks) = (QuantValue::Q4S, 4, 4, 8, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, so an f16 scale would be exact too.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // A region sits inside one block, and the packed line is one word of it.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 2), (KI, factor)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    packed_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            2,
            // One scale per `(row, block)`, two blocks a read: the width lands on `KB`.
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The q4 matmul with the weights on the right: what the shipped quant matmul does. One column
/// block per packed line, so a line never straddles a scale.
#[test]
fn a_packed_rhs_contracts_against_its_scales() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q4S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    // Two packed lines wide; `bn` says how many columns share a scale, and a line takes one.
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            Levels::leaf(&[(M, rows), (NB, blocks_n), (NI, bn), (KB, 1), (KI, block_k)])
                .walk_every(&[M, NB, NI, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    packed_matmul_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The same over 8-bit fields.
#[test]
fn an_eight_bit_packed_rhs_contracts_against_its_scales() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            Levels::leaf(&[(M, rows), (NB, blocks_n), (NI, bn), (KB, 1), (KI, block_k)])
                .walk_every(&[M, NB, NI, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    packed_matmul_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A block covering both lines: several lines share a scale, the direction that is always sound.
/// The other one (a block narrower than the line reading it) is refused by the contraction
/// (`mm_scaled: scale blocks must cover whole lines`), a comptime kernel panic on a worker thread.
#[test]
fn several_lines_may_share_one_scale() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 4, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    // Two packed lines wide, and one scale spanning both of them.
    let (cols, bn) = (factor * 2, factor * 2);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            Levels::leaf(&[(M, rows), (NB, blocks_n), (NI, bn), (KB, 1), (KI, block_k)])
                .walk_every(&[M, NB, NI, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    packed_matmul_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The native store needs no engine feature.** `i8` weights, their scales beside them, one
/// contraction: the tile serves the element its binding names and the block casts it, so an
/// `i8` store needs no packing stated.
#[test]
fn an_i8_operand_contracts_against_its_scales() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        ValidationResult::Skipped("backend has no native i8".to_string()).enforce();
        return;
    }

    // Cycling through every 8-bit signed value, sign extension included.
    let w: Vec<i32> = (0..rows * depth).map(|i| -128 + (i as i32 % 256)).collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let dtype = f32::elem_type_native();
    let (w_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(i8::elem_type_native())
        .custom(w.iter().map(|&v| v as f32).collect::<Vec<_>>())
        .generate_with_f32_host_data();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    native_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            i8::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The q4 decode gemv, end to end.** A packed weight tensor read in place, its scales as their
/// own operand, one row of activations, `N` across cubes, and the partials in registers for the
/// whole `K` walk: schemeless packed values, a scales operand, the rhs side, a promoted block.
#[test]
fn a_packed_decode_gemv_runs_in_this_spelling() {
    let (field, block_k, blocks_k) = (QuantValue::Q4S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    // Two cubes, each owning one packed line of columns.
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, 1),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            Levels::leaf(&[(NB, 1), (KB, 1)])
                .walk_every(&[KB])
                .cubes(&[NB])
                .build(),
        ),
        Form::Static,
    );

    packed_gemv::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| x[k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}

/// The same over 8-bit fields, so it runs on a device whose vectors cap at four.
#[test]
fn an_eight_bit_decode_gemv_runs_in_this_spelling() {
    let (field, block_k, blocks_k) = (QuantValue::Q8S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    let (cols, bn) = (factor * 2, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![blocks_k, blocks_n])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, 1),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            Levels::leaf(&[(NB, 1), (KB, 1)])
                .walk_every(&[KB])
                .cubes(&[NB])
                .build(),
        ),
        Form::Static,
    );

    packed_gemv::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            // One scale per `(block of K, block of N)`: `KI` is carried and addressed by
            // nothing, and the position inside a column block is not an axis of this operand at
            // all, which leaves its innermost axis one it actually varies over.
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| x[k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}

/// [`packed_gemv`] with the scales taken away: a packed rhs contracting into a promoted
/// accumulator through the plain `mm`.
#[cube(launch)]
fn packed_gemv_unscaled<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let w = w.tile(&space);
    let c = c.tile(&space);
    for cube in space {
        let x = x.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        let acc = c.accumulator::<E>(
            &x,
            &w,
            comptime!(Instruction::Registers {
                config: REGISTER_BLOCK
            }),
            Semiring::SUM_PROD,
        );
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma(&x.at(&step), &w.at(&step));
        }
        acc.drained_into(&c);
    }
}

/// A packed rhs drains from a promoted accumulator, exactly as its scaled twin does.
///
/// [`a_packed_decode_gemv_runs_in_this_spelling`] runs this block with the scales folded in, and
/// the two reach the same `block::contract` through the same `RegisterData`, so if a packed rhs
/// were what a promoted accumulator could not drain, the gemv could not be spelled either.
///
/// What makes both work is the accumulator being declared at the *served* width (`V = factor`); a
/// narrower one is refused by the width assert, and that is the only thing packing owes here.
#[test]
fn a_packed_rhs_drains_from_a_promoted_accumulator() {
    let (field, block_k, blocks_k) = (QuantValue::Q8S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    // Two cubes, each owning one packed line of columns.
    let (cols, bn) = (factor * 2, factor);

    let x: Vec<f32> = (0..depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, cols), (KB, blocks_k), (KI, block_k)]),
            Levels::leaf(&[(N, bn), (KB, 1)])
                .walk_every(&[KB])
                .cubes(&[N])
                .build(),
        ),
        Form::Static,
    );

    packed_gemv_unscaled::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth).map(|k| x[k] * w[k * cols + n] as f32).sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at col {n}: got {have}, want {want}"
        );
    }
}

/// Pack bytes four to a word, low byte first.
fn words_of_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &b)| acc | ((b as u32) << (j * 8)))
        })
        .collect()
}

/// **An 8-bit float field unpacks on read.** `e4m3` and `e5m2` codes four to a word, decoded
/// through the format rather than sign-extended.
#[test]
fn fp8_fields_unpack_on_read() {
    let client = cubecl::test_device().client();
    for value in [QuantValue::E4M3, QuantValue::E5M2] {
        check_fp8_codes_unpack(&client, value, Field::from(value));
    }
}

/// The same codes placed from their bits as `f16` and lifted, the decode a device that emulates
/// the format's conversion reads, held to every code on each device that reads them placed.
/// Placed, a small code is an `f16` subnormal (e4m3's 2^-9 is 2^-17 before its lift), which a
/// device may flush to zero, as CI's software Vulkan does; such a device converts the codes
/// instead ([`Field::read_on`]).
#[test]
fn fp8_codes_unpack_placed() {
    let client = cubecl::test_device().client();
    for value in [QuantValue::E4M3, QuantValue::E5M2] {
        let placed = Field::Placed {
            value,
            lifted: true,
        };
        if Field::from(value).read_on(&client) != placed {
            ValidationResult::Skipped(format!(
                "{value:?}: this device converts the codes, it never reads them placed"
            ))
            .enforce();
            continue;
        }
        check_fp8_codes_unpack(&client, value, placed);
    }
}

/// The same codes as an operand bound at launch reads them on this device ([`Field::read_on`]):
/// placed from their bits where it would emulate the conversion, through the format's decoder
/// elsewhere.
#[test]
fn fp8_codes_unpack_as_the_device_reads_them() {
    let client = cubecl::test_device().client();
    for value in [QuantValue::E4M3, QuantValue::E5M2] {
        check_fp8_codes_unpack(&client, value, Field::from(value).read_on(&client));
    }
}

/// Every `value` code but its NaNs, which have nothing to compare against, packed four to a word,
/// unpacked through `field` and checked bit for bit against the format's own decode.
fn check_fp8_codes_unpack(client: &Client, value: QuantValue, field: Field) {
    let (rows, cols) = (16, 16);
    let factor = 32 / field.size_bits();

    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let decode = |bits: u8| match value {
        QuantValue::E4M3 => e4m3::from_bits(bits).to_f32(),
        QuantValue::E5M2 => e5m2::from_bits(bits).to_f32(),
        other => panic!("check_fp8_codes_unpack: {other:?} is not an 8-bit float"),
    };
    // Every code once, a NaN code read as zero instead.
    let bytes: Vec<u8> = (0..=u8::MAX)
        .map(|bits| if decode(bits).is_nan() { 0 } else { bits })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words_of_bytes(&bytes))),
        u32::elem_type_native(),
    );
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    packed_copy::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(field),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        uncut(client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = decode(bytes[m * cols + n]);
            let have = got.get_f32(&[m, n]);
            assert!(
                have.to_bits() == want.to_bits(),
                "{field:?} at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **Scales stored as `ue8m0` bytes are read in their own width.** Four codes to a word, the
/// scales operand bound one word wide and served one scale at a time ([`TileSpec::packed`]), on
/// the lhs of the memory-backed leaf, which steps the block index at runtime.
///
/// A `ue8m0` code is a bare exponent, so the scales are powers of two and the answer is exact.
#[test]
fn ue8m0_scales_are_read_as_bytes() {
    check_ue8m0_scales(8, 4);
}

fn check_ue8m0_scales(block: usize, blocks: usize) {
    let (field, rows, cols) = (QuantValue::Q8S, 4, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Exponents -2..=2: the code is the exponent biased by 127.
    let exponents: Vec<i32> = (0..rows * blocks).map(|i| (i % 5) as i32 - 2).collect();
    let s: Vec<f32> = exponents.iter().map(|&e| 2f32.powi(e)).collect();
    let codes: Vec<u8> = exponents.iter().map(|&e| (e + 127) as u8).collect();

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    // Shape and strides count scales; a row of `blocks` codes is one word.
    let s_tensor = TensorHandle::new_contiguous(
        vec![rows, blocks],
        client.create(Bytes::from_elems(words_of_bytes(&codes))),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            // Four scales a word, so a region is the four blocks one read of them covers.
            Levels::leaf(&[(M, rows), (N, cols), (KB, 4), (KI, factor)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    let w_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    packed_matmul_byte_scales::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(w_projection.clone()).packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(w_projection.scales_per(KB)).packed(Fp8Format::UE8M0),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **Scales stored at their own float width are read as fields of a word too.** Two `f16`
/// scales share a word, so a row of them need not start on one and the slot is picked at the
/// read, exactly as a byte field's is.
#[test]
fn half_scales_are_read_as_fields_of_a_word() {
    check_float_scales(FloatKind::F16, 16, 2);
}

/// An `f32` scale fills the word it is stored in, which is the case with no slot to find: the
/// read is the word, reinterpreted.
#[test]
fn full_width_scales_fill_the_word_they_are_read_from() {
    check_float_scales(FloatKind::F32, 8, 4);
}

/// [`check_ue8m0_scales`] with the scales stored as whole floats of `kind`. Halves and quarters,
/// which every width here holds exactly.
fn check_float_scales(kind: FloatKind, block: usize, blocks: usize) {
    let (field, rows, cols) = (QuantValue::Q8S, 4, 4);
    let depth = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * depth)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let x: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks)
        .map(|i| 0.25 * (1 + i % 4) as f32)
        .collect();
    let scale_words = match kind {
        FloatKind::F32 => s.iter().map(|v| v.to_bits()).collect(),
        _ => s
            .chunks(2)
            .map(|pair| {
                pair.iter().enumerate().fold(0u32, |acc, (j, &v)| {
                    acc | ((f16::from_f32(v).to_bits() as u32) << (j * 16))
                })
            })
            .collect::<Vec<u32>>(),
    };

    let dtype = f32::elem_type_native();
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, depth],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    // Shape and strides count scales; how many share a word is the field's to say.
    let s_tensor = TensorHandle::new_contiguous(
        vec![rows, blocks],
        client.create(Bytes::from_elems(scale_words)),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            // A region is the blocks one word of scales covers.
            Levels::leaf(&[
                (M, rows),
                (N, cols),
                (KB, Field::of_float(kind).per_word()),
                (KI, factor),
            ])
            .walk_every(&[M, N, KB, KI])
            .build(),
        ),
        Form::Static,
    );

    let w_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    packed_matmul_byte_scales::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(w_projection.clone()).packed(field),
        ),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(w_projection.scales_per(KB)).packed(kind),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| w[m * depth + k] as f32 * s[m * blocks + k / block] * x[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **Byte scales reach the promoted block on the rhs.** `e4m3` codes for the decode gemv's
/// scales, one per `(block of K, block of N)`, read one at a time under the block's constant
/// column ordinal. Halves, which `e4m3` holds exactly.
#[test]
fn e4m3_scales_reach_the_promoted_block() {
    let (field, block_k, blocks_k) = (QuantValue::Q8S, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    // Four packed lines of columns, so a row of the scales is one word of four codes.
    let (cols, bn) = (factor * 4, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();
    let codes: Vec<u8> = s.iter().map(|&v| e4m3::from_f32(v).to_bits()).collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let s_tensor = TensorHandle::new_contiguous(
        vec![blocks_k, blocks_n],
        client.create(Bytes::from_elems(words_of_bytes(&codes))),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, 1),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            // One read of the scales is four column blocks, so one cube owns all four.
            Levels::leaf(&[(NB, 4), (KB, 1)])
                .walk_every(&[KB])
                .cubes(&[NB])
                .build(),
        ),
        Form::Static,
    );

    packed_gemv_byte_scales::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            ))
            .packed(QuantValue::E4M3),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| x[k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}

/// **A packed rhs with byte scales reaches the tensor cores.** Eight activation rows against a
/// `q8` weight packed along its columns, `e4m3` scales per `(K block, N block)` read one at a time,
/// the weight unpacked and scaled into the `8x8x8` `B` fragment. Prefill: one fragment a region.
///
/// The kernel that dequantizes the weight itself before the multiply must match the leaf's decode
/// bit for bit: the explicit call is the same decode, placed where the kernel can name its element.
#[test]
fn a_packed_rhs_reaches_the_tensor_cores() {
    let (field, rows, block_k, blocks_k) = (QuantValue::Q8S, 8, 8, 4);
    let depth = block_k * blocks_k;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    // Four column blocks of one packed line each; a fragment covers two of them.
    let (cols, bn) = (factor * 4, factor);
    let blocks_n = cols / bn;

    let x: Vec<f32> = (0..rows * depth).map(|i| (i % 7) as f32 - 3.0).collect();
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..depth * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let s: Vec<f32> = (0..blocks_k * blocks_n)
        .map(|i| (i as f32 + 1.0) / 2.0)
        .collect();
    // Halves, two to a word: one read of the scales is the region's two column blocks, and the
    // landing builds both blocks' lines under it.
    let scale_words: Vec<u32> = s
        .chunks(2)
        .map(|pair| {
            (f16::from_f32(pair[0]).to_bits() as u32)
                | ((f16::from_f32(pair[1]).to_bits() as u32) << 16)
        })
        .collect();

    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let w_tensor = TensorHandle::new_contiguous(
        vec![depth, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let s_tensor = TensorHandle::new_contiguous(
        vec![blocks_k, blocks_n],
        client.create(Bytes::from_elems(scale_words)),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (M, rows),
                (NB, blocks_n),
                (NI, bn),
                (KB, blocks_k),
                (KI, block_k),
            ]),
            Levels::leaf(&[(M, rows), (NB, 8 / bn), (NI, bn), (KB, 1), (KI, block_k)])
                .walk_every(&[M, NB, NI, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    let x_arg = || {
        TileArgLaunch::new(
            x_tensor.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                ],
            )),
        )
    };
    let w_arg = || {
        TileArgLaunch::new(
            w_tensor.clone().binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB, NI],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block_k), (KI, 1)]),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            ))
            .packed(field),
        )
    };
    let s_arg = || {
        TileArgLaunch::new(
            s_tensor.clone().binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, NB],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(NB)],
            ))
            .packed(FloatKind::F16),
        )
    };
    let out_arg = |out: &TensorHandle| {
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, bn), (NI, 1)]),
                ],
            )),
        )
    };

    packed_cmma_rhs::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        x_arg(),
        w_arg(),
        s_arg(),
        out_arg(&c),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    x[m * depth + k] * w[k * cols + n] as f32 * s[(k / block_k) * blocks_n + n / bn]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }

    let dequantized = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    packed_cmma_rhs_dequantized::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        x_arg(),
        w_arg(),
        s_arg(),
        out_arg(&dequantized),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
        dtype,
    );
    let explicit = HostData::from_tensor_handle(&client, dequantized, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            assert_eq!(
                explicit.get_f32(&[m, n]),
                got.get_f32(&[m, n]),
                "at ({m}, {n}): the explicit dequantize differs from the leaf's"
            );
        }
    }
}

/// `output = unpack(input) ⊗ scales`: the decode a kernel states by copying its scaled values.
#[cube(launch)]
fn packed_scaled_copy(
    input: &TileArg<'_>,
    scales: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
) {
    let input = input.tile(&space);
    let scales = scales.tile(&space);
    let mut output = output.tile(&space);
    output.dequantize_from(&input.mul(&scales));
}

/// Eight-bit fields and one scale per block of sixteen, decoded by the copy that states them: the
/// words unpack and every value takes its block's scale.
#[test]
fn a_copy_of_scaled_packed_fields_decodes_them() {
    let (field, rows, block, blocks) = (QuantValue::Q8S, 8, 16, 2);
    let cols = block * blocks;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let scales: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 4.0).collect();
    let (scales_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(f32::elem_type_native())
        .custom(scales.clone())
        .generate_with_f32_host_data();

    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (KB, blocks), (KI, block)]);
    let projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    packed_scaled_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(projection.clone()).packed(field),
        ),
        TileArgLaunch::new(
            scales_t.binding().into_tensor_arg(),
            f32::elem_type_native(),
            1,
            TileSpec::new(projection.scales_per(KB)),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::new(projection),
        ),
        uncut(&client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = values[m * cols + n] as f32 * scales[m * blocks + n / block];
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The axis of a codebook's table: its positions, outside the kernel's space.
const TBL: Axis = Axis(6);

/// `output = table[input] ⊗ scales`, the input bound under its codebook and its scales.
#[cube(launch)]
fn looked_up_copy(input: &TileArg<'_>, output: &TileArg<'_>, space: Partitioning) {
    let input = input.tile(&space);
    let mut output = output.tile(&space);
    output.dequantize_from(&input);
}

/// Eight-bit indices into a table of 256 entries, scaled per block of sixteen: each field names
/// its entry, raw and unsigned (an index past 127 must not read as negative), and the entry takes
/// its block's scale.
#[test]
fn a_copy_of_looked_up_fields_decodes_them() {
    check_looked_up_copy(8, 4);
}

/// Four- and two-bit indices copied into lines narrower than their word, the read a vec4 device
/// gives a low-bit cache: each word's fields unpack a destination line at a time.
#[test]
fn a_copy_of_looked_up_fields_narrower_than_a_word_decodes_them() {
    check_looked_up_copy(4, 4);
    check_looked_up_copy(2, 4);
}

/// `table[index] ⊗ scale` for `bits`-wide indices, copied into lines `width` wide.
fn check_looked_up_copy(bits: usize, width: usize) {
    let (rows, block, blocks) = (8, 16, 2);
    let cols = block * blocks;
    let factor = 32 / bits;
    let entries = 1usize << bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if width > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {width}-wide line"
        ))
        .enforce();
        return;
    }

    let indices: Vec<u32> = (0..rows * cols)
        .map(|i| ((i * 37 + 3) % entries) as u32)
        .collect();
    let words: Vec<u32> = indices
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | (v << (j * bits)))
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let table: Vec<f32> = (0..entries).map(|i| (i as f32 - 127.5) / 3.0).collect();
    let (table_t, _) = TestInput::builder(client.clone(), shape![entries])
        .dtype(f32::elem_type_native())
        .custom(table.clone())
        .generate_with_f32_host_data();
    let scales: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 4.0).collect();
    let (scales_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(f32::elem_type_native())
        .custom(scales.clone())
        .generate_with_f32_host_data();
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (KB, blocks), (KI, block)]);
    let projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    let input = TileArgLaunch::new(
        input.binding().into_tensor_arg(),
        u32::elem_type_native(),
        1,
        TileSpec::new(projection.clone()).packed(Field::Index { bits }),
    )
    .codebook(TileArgLaunch::new(
        table_t.binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        TileSpec::direct(&[TBL]),
    ))
    .expect("a plain table of an entry per index")
    .scales(TileArgLaunch::new(
        scales_t.binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        TileSpec::new(projection.scales_per(KB)),
    ));
    looked_up_copy::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        input,
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            width,
            TileSpec::new(projection),
        ),
        uncut(&client, &space, &space).partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = table[indices[m * cols + n] as usize] * scales[m * blocks + n / block];
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// How [`decoded_lines`] reads each line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum LineRead {
    /// Masked where it overhangs the operand ([`Decoded::read`](cubecl_tile::kind::Decoded::read)).
    Masked,
    /// Untested, the kernel having tested it ([`Decoded::read_unchecked`](cubecl_tile::kind::Decoded::read_unchecked)).
    Untested,
}

/// Every line of `input`, `lines` a row, read into registers decoded, a unit a line, and stored
/// in `output`.
#[cube(launch)]
fn decoded_lines<N: Size>(
    input: &TileArg<'_>,
    output: &mut Tensor<Vector<f32, N>>,
    space: Partitioning,
    #[comptime] lines: usize,
    #[comptime] read: LineRead,
) {
    let input = input.tile(&space);
    let decoded = input.decoded::<f32, N>();
    let at = ABSOLUTE_POS;
    let (row, line) = (at / lines, at % lines);
    if at < output.len() {
        output[at] = match comptime!(read) {
            LineRead::Masked => decoded.read(row, line),
            LineRead::Untested => decoded.read_unchecked(row, line),
        };
    }
}

/// **A unit reads looked-up fields into its registers decoded.** Each `bits`-wide field names its
/// entry of the table, and the row's scale multiplies it, as a copy decodes them, with no stage to
/// decode into: a line inside one word reads the fields it needs of it, a wider one every word it
/// spans (16-bit indices, two a word, four a line, on a device whose vectors stop at four).
#[test]
fn a_decoded_read_of_looked_up_fields_matches_the_copy() {
    for (bits, width) in [(8, 4), (4, 4), (2, 4), (4, 2), (16, 4), (8, 8), (4, 16)] {
        check_decoded_lines(bits, width, LineRead::Masked);
    }
}

/// **A line the kernel tested itself reads the same untested.** The read skips the bounds test
/// it would repeat, and decodes the same values.
#[test]
fn an_untested_decoded_read_matches_the_masked_one() {
    for (bits, width) in [(8, 4), (2, 4), (16, 4)] {
        check_decoded_lines(bits, width, LineRead::Untested);
    }
}

/// **A unit reads integer fields into its registers decoded, with no table.** Signed 4-bit values,
/// sign extension included, each under its row's scale, read two a line, a half of a word's
/// fields at each read.
#[test]
fn a_decoded_read_of_integer_fields_scales_them() {
    let (field, rows, cols, width) = (QuantValue::Q4S, 4, 32, 2);
    let bits = field.size_bits();
    let factor = 32 / bits;
    let client = cubecl::test_device().client();

    let span = 1i32 << bits;
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + ((i * 5) as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter().enumerate().fold(0u32, |acc, (j, &v)| {
                acc | (((v as u32) & mask) << (j * bits))
            })
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let scales: Vec<f32> = (0..rows).map(|i| (i as f32 + 1.0) / 4.0).collect();
    let (scales_t, _) = TestInput::builder(client.clone(), shape![rows])
        .dtype(f32::elem_type_native())
        .custom(scales.clone())
        .generate_with_f32_host_data();
    let lines = rows * cols / width;
    let output = TestInput::builder(client.clone(), shape![lines * width])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    let input = TileArgLaunch::new(
        input.binding().into_tensor_arg(),
        u32::elem_type_native(),
        1,
        TileSpec::direct(&[M, N]).packed(field),
    )
    .scales(TileArgLaunch::new(
        scales_t.binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        TileSpec::direct(&[M]),
    ));
    decoded_lines::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(lines as u32),
        width,
        input,
        output.clone().binding().into_tensor_arg(),
        uncut(&client, &space, &space).partitioning_arg(),
        cols / width,
        LineRead::Masked,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = values[m * cols + n] as f32 * scales[m];
            let have = got.get_f32(&[m * cols + n]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// `table[index] ⊗ scale[row]` for `bits`-wide indices, read `width` values a line as `read` says.
fn check_decoded_lines(bits: usize, width: usize, read: LineRead) {
    let (rows, cols) = (4, 64);
    let factor = 32 / bits;
    let entries = 1usize << bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if width > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {width}-wide line"
        ))
        .enforce();
        return;
    }

    let indices: Vec<u32> = (0..rows * cols)
        .map(|i| ((i * 37 + 3) % entries) as u32)
        .collect();
    let words: Vec<u32> = indices
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | (v << (j * bits)))
        })
        .collect();
    // A packed buffer is stated in the values its words hold.
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let table: Vec<f32> = (0..entries).map(|i| (i as f32 - 127.5) / 3.0).collect();
    let (table_t, _) = TestInput::builder(client.clone(), shape![entries])
        .dtype(f32::elem_type_native())
        .custom(table.clone())
        .generate_with_f32_host_data();
    let scales: Vec<f32> = (0..rows).map(|i| (i as f32 + 1.0) / 4.0).collect();
    let (scales_t, _) = TestInput::builder(client.clone(), shape![rows])
        .dtype(f32::elem_type_native())
        .custom(scales.clone())
        .generate_with_f32_host_data();
    let lines = rows * cols / width;
    let output = TestInput::builder(client.clone(), shape![lines * width])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, rows), (N, cols)]);
    let input = TileArgLaunch::new(
        input.binding().into_tensor_arg(),
        u32::elem_type_native(),
        1,
        TileSpec::direct(&[M, N]).packed(Field::Index { bits }),
    )
    .codebook(TileArgLaunch::new(
        table_t.binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        TileSpec::direct(&[TBL]),
    ))
    .expect("a plain table of an entry per index")
    .scales(TileArgLaunch::new(
        scales_t.binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        TileSpec::direct(&[M]),
    ));
    decoded_lines::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(lines as u32),
        width,
        input,
        output.clone().binding().into_tensor_arg(),
        uncut(&client, &space, &space).partitioning_arg(),
        cols / width,
        read,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want = table[indices[m * cols + n] as usize] * scales[m];
            let have = got.get_f32(&[m * cols + n]);
            assert!(
                (have - want).abs() < 1e-6,
                "{bits}-bit fields read {width} a line, at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **A packed operand is bound at the words its buffer holds.** At any other element its buffer
/// would be read as values it does not hold, so the binding refuses it.
#[test]
#[should_panic(
    expected = "TileArgLaunch: a packed operand is bound at the u32 words its buffer \
                           holds, not"
)]
fn a_packed_operand_is_bound_at_its_words() {
    let client = cubecl::test_device().client();
    let words = TensorHandle::new_contiguous(
        vec![1, 1],
        client.create_from_slice(u32::as_bytes(&[0])),
        u32::elem_type_native(),
    );
    TileArgLaunch::new(
        words.binding().into_tensor_arg(),
        f16::elem_type_native(),
        1,
        TileSpec::direct(&[M, N]).packed(Field::Quant(QuantValue::E2M1)),
    );
}

/// A word-wide float field staged once for a walk at `decoded`, as an attention stages its query,
/// and the stage copied out as it holds it.
#[cube(launch)]
fn word_field_staged_for_a_walk(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] decoded: Option<cubecl::ir::ElemType>,
) {
    let input = input.tile(&space);
    let output = output.tile(&space);
    let walk = space.walk();
    let first = walk.region(0);
    let source = input.at(&first);
    let mut staged = source.stage_for(&walk, StageStorage::Strided, decoded);
    staged.copy_from(&source);
    sync_cube();
    let mut out = output.at(&first);
    out.copy_cast_from(&staged);
}

/// `f32` values bound as their words, one a word, staged for a walk at `decoded` and copied out.
fn staged_for_a_walk(values: &[f32], decoded: Option<cubecl::ir::ElemType>) -> Vec<f32> {
    let (rows, cols) = (8, 16);
    let client = cubecl::test_device().client();
    let words: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols)]),
            Levels::leaf(&[(M, rows), (N, cols)])
                .walk_every(&[M])
                .build(),
        ),
        Form::Static,
    );
    word_field_staged_for_a_walk::launch(
        &client,
        CubeCount::new_single(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(Field::Float(FloatKind::F32)),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        decoded,
    );
    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    (0..rows * cols)
        .map(|i| got.get_f32(&[i / cols, i % cols]))
        .collect()
}

/// **A stage for a walk holds a word-wide float field at the element its kernel names.** Values
/// finer than `f16` keeps come back rounded to `f16` from a stage named `f16`, and exact from one
/// named none, which holds `f32`.
#[test]
fn a_word_wide_field_staged_for_a_walk_holds_the_element_its_kernel_names() {
    let values: Vec<f32> = (0..8 * 16)
        .map(|i| 1.0 + i as f32 * 2f32.powi(-12))
        .collect();
    let rounded: Vec<f32> = values
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect();
    assert_ne!(rounded, values, "the values must be finer than f16 keeps");
    let f16 = cubecl::ir::ElemType::Float(FloatKind::F16);
    assert_eq!(staged_for_a_walk(&values, Some(f16)), rounded);
    assert_eq!(staged_for_a_walk(&values, None), values);
}
