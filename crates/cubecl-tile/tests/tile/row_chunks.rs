//! A stage whose block rows are swizzled ([`RowChunks::Swizzled`]) or padded
//! ([`RowChunks::Padded`]) holds the same product as one whose rows lie in order: filled whole,
//! filled through registers, and read by the instruction the accumulator picks for the leaf: the
//! manual mma transport (through `ldmatrix` where the device has it) or, in order or padded, the
//! cmma one where the device offers the leaf's shape — a row-major rhs and a weight stored
//! `{n, k}` alike. A swizzled stage only the manual mma reads, so where the accumulator picks cmma
//! for the leaf (Metal's `8×8×8`) its case reports the test skipped. A global window is read
//! manually.
//!
//! In `f16` (and `bf16` through `ldmatrix`) summed in `f32`, at `16×16×16` where the device offers
//! it and at the shape it does otherwise (NVIDIA's manual mma is `16×8×16`); a device with no such
//! instruction, or no `ldmatrix` for the element, reports the test skipped. Lines of
//! other byte widths (a 4-byte element's, a packed word's) are arranged by the same rule, held on
//! the host by `RowArrangement`'s and `ChunkSwizzle`'s own tests.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{ir::ElemType, prelude::*, zspace::shape};
use cubecl_tile::stage::{Prefetch, RowChunks};
use cubecl_tile::*;

use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Where the fragments are read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Staging {
    /// A shared-memory stage, filled a step of `K` at a time.
    Stage,
    /// The same stage, with a 4-byte shared allocation declared before it: at the element's own
    /// alignment the stage's rows would start off 16 bytes.
    StageAfterSmallAllocation,
    /// No stage: the fragments are read straight out of the global tiles.
    Global,
}

/// Which instruction's shapes the leaf is cut to: the accumulator picks the instruction for that
/// shape, the stage's arrangement and the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Leaf {
    /// A shape the device offers through the manual mma.
    Mma,
    /// A shape the device offers through the vendor's fragment API.
    Cmma,
}

/// `c = a · b` over a walk along `K` whose every region is one stage of `outer`'s tile, read a
/// fragment of `inner` at a time through the instruction the accumulator picks, the stage laid
/// out as `storage`.
#[cube(launch)]
fn staged_k_walk<EA: Numeric>(
    a: &TileTensor<'_>,
    b: &TileTensor<'_>,
    c: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Prefetch,
    #[comptime] staging: Staging,
    #[define(EA)] _sum: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    let mut acc = c.accumulator::<EA>(&a, &b, Semiring::SUM_PROD);
    let walk = space.over(&outer);
    if comptime!(staging == Staging::Global) {
        for region in walk {
            let acc_o = acc.at(&region);
            for fragment in region.over(&inner).unrolled() {
                let mut acc_f = acc_o.at(&fragment);
                acc_f.mma(&a.at(&fragment), &b.at(&fragment));
            }
        }
    } else {
        staged(
            &a,
            &b,
            &mut acc,
            walk,
            inner.clone(),
            storage,
            depth,
            schedule,
            staging,
        );
    }
    acc.drained_into(&c);
}

/// The walk through a shared-memory stage, `staging` saying what is declared before it.
#[cube]
fn staged(
    a: &Tile,
    b: &Tile,
    acc: &mut Tile,
    walk: Walk,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Prefetch,
    #[comptime] staging: Staging,
) {
    if comptime!(staging == Staging::StageAfterSmallAllocation) {
        // Written before the stage is declared and read after the walk, so the allocation lives
        // across the stage and cannot be dropped: an unread write would leave the stage at 0.
        let mut small = Shared::<[u32]>::new_slice(1usize);
        if UNIT_POS == 0 {
            small[0] = 1u32;
        }
        sync_cube();
        walk_stages(a, b, acc, walk, inner, storage, depth, schedule);
        sync_cube();
        if small[0] != 1u32 {
            acc.zero();
        }
    } else {
        walk_stages(a, b, acc, walk, inner, storage, depth, schedule);
    }
}

/// `a · b` into `acc`, a stage of `storage` at a time.
#[cube]
fn walk_stages(
    a: &Tile,
    b: &Tile,
    acc: &mut Tile,
    walk: Walk,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Prefetch,
) {
    let mut stages = Stages::smem(&walk, a, b, storage, depth, None);
    stages.walk(walk, schedule, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for fragment in region.over(&inner).unrolled() {
                let mut acc_f = acc_o.at(&fragment);
                acc_f.mma(&a_s.at(&fragment), &b_s.at(&fragment));
            }
        });
    });
}

/// The `m × n × k` shape this device offers `leaf` at in `input` summed in `f32`: `16×16×16` where
/// it does, else the shallowest other at least `line_depth` deep, else the shallowest, so the
/// stages are read by the instruction each device has. A case whose lines run along `K` asks for a
/// line's depth: a step must start on a line, and NVIDIA lists its `16×8×8` mma before its
/// `16×8×16` one.
/// Reported rather than silently passed where the device has none. The manual mma loads through
/// `ldmatrix` where the device has it for `input`, and addresses each unit's lines itself
/// otherwise.
fn instruction_shape(
    client: &cubecl::client::Client,
    leaf: Leaf,
    input: ElemType,
    line_depth: usize,
) -> Option<(usize, usize, usize)> {
    let f32 = f32::elem_type_native();
    let matmul = &client.properties().features.matmul;
    let configs = match leaf {
        Leaf::Mma => &matmul.mma,
        Leaf::Cmma => &matmul.cmma,
    };
    let shapes: Vec<(usize, usize, usize)> = configs
        .iter()
        .filter(|cfg| cfg.a_type == input && cfg.b_type == input && cfg.cd_type == f32)
        .map(|cfg| (cfg.m as usize, cfg.n as usize, cfg.k as usize))
        .collect();
    let shape = shapes
        .iter()
        .copied()
        .find(|&shape| shape == (16, 16, 16))
        .or_else(|| shapes.iter().copied().find(|&(_, _, k)| k >= line_depth))
        .or_else(|| shapes.first().copied());
    if shape.is_none() {
        ValidationResult::Skipped(format!("device has no {input:?} {leaf:?} summed in f32"))
            .enforce();
    }
    shape
}

/// Small integers, exact in `f16`, and differing between neighbours along every axis.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + seed) % 13) as f32 - 6.0)
        .collect()
}

/// One product through [`staged_k_walk`].
#[derive(Clone, Copy, Debug)]
struct Case {
    chunks: RowChunks,
    /// The problem's `M` and `N`, which one stage holds whole: past the block's 16, a stage holds
    /// several blocks along each.
    mn: usize,
    /// How much of `K` one stage holds.
    stage_k: usize,
    /// The width the operands are served in.
    v: usize,
    /// The rhs stored `{n, k}` rather than `{k, n}`.
    transposed: bool,
    /// The lhs stored `{k, m}` rather than `{m, k}`.
    lhs_transposed: bool,
    /// Stages in flight.
    depth: usize,
    schedule: Prefetch,
    leaf: Leaf,
    /// Where the fragments are read from.
    staging: Staging,
    /// The operands' element, `f16` or `bf16`, summed in `f32`.
    input: ElemType,
}

impl Case {
    /// A 64-byte row (32 `f16` of `K`) in 8-wide lines, the rhs row-major, one stage, read by the
    /// manual mma transport.
    const DEFAULT: Self = Self {
        chunks: RowChunks::InOrder,
        mn: 16,
        stage_k: 32,
        v: 8,
        transposed: false,
        lhs_transposed: false,
        depth: 1,
        schedule: Prefetch::InSlots,
        leaf: Leaf::Mma,
        staging: Staging::Stage,
        input: ElemType::Float(cubecl::ir::FloatKind::F16),
    };
}

fn check(case: Case) {
    let Case {
        chunks,
        mn,
        stage_k,
        v,
        transposed,
        lhs_transposed,
        depth,
        schedule,
        leaf,
        staging,
        input,
    } = case;
    let client = cubecl::test_device().client();
    // A `{m, k}` lhs or a `{n, k}` rhs runs its lines along `K`.
    let lines_along_k = !lhs_transposed || transposed;
    let line_depth = if lines_along_k { v } else { 1 };
    let Some((edge_m, edge_n, edge_k)) = instruction_shape(&client, leaf, input, line_depth) else {
        return;
    };
    // A drained sum picks its instruction before any stage exists, so where the fragment API offers
    // the leaf (Metal's `8×8×8`) it reads the stage off a pointer and a stride, which a swizzled
    // row is not: the kernel refuses as it expands. Reported, not run.
    let picked = Instruction::for_accumulator(
        client.properties(),
        client.target_properties(),
        space::ComputeScope::Plane,
        FragmentSource::Values(input),
        FragmentSource::Values(input),
        f32::elem_type_native(),
        [edge_m, edge_n, edge_k],
    );
    if chunks == RowChunks::Swizzled && picked == Some(Instruction::Cmma) {
        ValidationResult::Skipped(format!(
            "the fragment API reads this device's {edge_m}×{edge_n}×{edge_k} leaf, and no \
             swizzled stage"
        ))
        .enforce();
        return;
    }
    let (m, n, k) = (mn, mn, 128usize);
    // A row-major rhs's lines run along `N`, and a fragment narrower than a line starts inside one,
    // which the manual mma load refuses: reported, not run.
    if leaf != Leaf::Cmma && !transposed && edge_n < v {
        ValidationResult::Skipped(format!(
            "a {v}-wide line runs past the {edge_n}-wide mma fragment along N"
        ))
        .enforce();
        return;
    }
    // The same for a `{k, m}` lhs, whose lines run along `M`.
    if leaf != Leaf::Cmma && lhs_transposed && edge_m < v {
        ValidationResult::Skipped(format!(
            "a {v}-wide line runs past the {edge_m}-wide mma fragment along M"
        ))
        .enforce();
        return;
    }
    // Lines along `K` and no instruction as deep as one would start a step inside a line, which the
    // descent refuses: an `f16` mma 8 deep and no deeper.
    if lines_along_k && edge_k < v {
        ValidationResult::Skipped(format!(
            "a {v}-wide line runs past the {edge_k}-deep instruction along K"
        ))
        .enforce();
        return;
    }
    assert!(
        stage_k.is_multiple_of(edge_k),
        "{case:?}: a stage {stage_k} deep is not whole {edge_k}-deep steps"
    );
    let a = values(m * k, 0);
    let b = values(k * n, 5);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum()
        })
        .collect();
    // A `{n, k}` weight holds the same logical `b`, laid down along `k`.
    let b_stored: Vec<f32> = match transposed {
        false => b.clone(),
        true => (0..n * k).map(|idx| b[(idx % k) * n + idx / k]).collect(),
    };
    // A `{k, m}` lhs holds the same logical `a`, laid down along `m`.
    let (a_stored, a_shape) = match lhs_transposed {
        false => (a.clone(), shape![m, k]),
        true => (
            (0..k * m).map(|idx| a[(idx % m) * k + idx / m]).collect(),
            shape![k, m],
        ),
    };
    let (a_handle, _) = TestInput::builder(client.clone(), a_shape)
        .dtype(input)
        .custom(a_stored)
        .generate_with_f32_host_data();
    let b_shape = if transposed {
        shape![n, k]
    } else {
        shape![k, n]
    };
    let (b_handle, _) = TestInput::builder(client.clone(), b_shape)
        .dtype(input)
        .custom(b_stored)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge_m), (N, edge_n), (K, edge_k)])
                .walk(&[(M, m / edge_m), (N, n / edge_n), (K, stage_k / edge_k)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    // A stage holds the whole of `M` and `N`, `stage_k` of `K`; a block is one fragment's rows,
    // `stage_k` deep, and the stage a grid of them wherever `mn` passes the fragment's edge.
    let storage = StageStorage::Tiled {
        block: vec![(M, edge_m), (N, edge_n), (K, stage_k)],
        chunks,
    };
    let a_axes: &'static [Axis] = if lhs_transposed { &[K, M] } else { &[M, K] };
    let b_axes: &'static [Axis] = if transposed { &[N, K] } else { &[K, N] };
    let bind = |binding, element, axes: &'static [Axis], width| {
        launcher
            .arg(binding, element)
            .axes(axes)
            .vectorize(width)
            .build()
            .unwrap()
    };
    staged_k_walk::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        bind(a_handle.clone().binding(), input, a_axes, v).arg(),
        bind(b_handle.clone().binding(), input, b_axes, v).arg(),
        bind(out.clone().binding(), f32::elem_type_native(), &[M, N], 1).arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        storage,
        depth,
        schedule,
        staging,
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    crate::support::assert_equals_approx(&got, &expected, 1e-3).enforce()
}

/// A 64-byte row (32 `f16` of `K`), the block the cmma transport conflicts on, swizzled or padded:
/// the same product as in order.
#[test]
fn an_arranged_stage_holds_the_product_of_one_in_order() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            ..Case::DEFAULT
        });
    }
}

/// A 32-byte row (16 `f16` of `K`), whose rows take two keys or pad by half their length.
#[test]
fn an_arranged_stage_one_instruction_deep_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            stage_k: 16,
            ..Case::DEFAULT
        });
    }
}

/// A 128-byte row (64 `f16` of `K`), four instructions deep.
#[test]
fn an_arranged_stage_four_instructions_deep_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                stage_k: 64,
                transposed,
                ..Case::DEFAULT
            });
        }
    }
}

/// Lines narrower than a chunk: four lines move together, or four pad a row.
#[test]
fn an_arranged_stage_of_narrow_lines_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            v: 2,
            ..Case::DEFAULT
        });
    }
}

/// A weight stored `{n, k}`, staged as it lies and read col-major, each lane's register vector a
/// run along its lines.
#[test]
fn a_weight_stored_along_k_is_read_where_it_lies() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            transposed: true,
            ..Case::DEFAULT
        });
    }
}

/// Two stages in flight, and the next held in registers across the contraction: both fills
/// write an arranged stage where its reads look.
#[test]
fn an_arranged_stage_is_filled_ahead_and_through_registers() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for schedule in [Prefetch::InSlots, Prefetch::InRegisters] {
            for transposed in [false, true] {
                check(Case {
                    chunks,
                    transposed,
                    depth: 2,
                    schedule,
                    ..Case::DEFAULT
                });
            }
        }
    }
}

/// The cmma transport reads a padded stage off its pointer and its stride, the row pitched past
/// its padding: the same product as in order, a row-major rhs and a weight stored `{n, k}` alike.
#[test]
fn the_fragment_api_reads_a_padded_stage() {
    for chunks in [RowChunks::InOrder, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                transposed,
                leaf: Leaf::Cmma,
                ..Case::DEFAULT
            });
        }
    }
}

/// A stage of two blocks along `M` and `N` each: a block's rows are pitched or swizzled within it,
/// and the next block starts past the last one's padding.
#[test]
fn an_arranged_stage_of_several_blocks_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            for schedule in [Prefetch::InSlots, Prefetch::InRegisters] {
                check(Case {
                    chunks,
                    mn: 32,
                    transposed,
                    schedule,
                    ..Case::DEFAULT
                });
            }
        }
    }
}

/// Lines wider than a chunk (16 `f16`, 32 bytes): a swizzle moves whole lines, and a pad is one
/// line long.
#[test]
fn an_arranged_stage_of_wide_lines_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                v: 16,
                stage_k: 64,
                transposed,
                ..Case::DEFAULT
            });
        }
    }
}

/// The cmma transport over a stage of several padded blocks, whose rows it reads off a pointer
/// and the pitched stride.
#[test]
fn the_fragment_api_reads_a_padded_stage_of_several_blocks() {
    for transposed in [false, true] {
        check(Case {
            chunks: RowChunks::Padded,
            mn: 32,
            transposed,
            leaf: Leaf::Cmma,
            ..Case::DEFAULT
        });
    }
}

/// The `ldmatrix` transport reads every arrangement a unit can address, each unit handing the
/// instruction its row where the stage arranged it: in order, padded, and swizzled, a row-major rhs
/// and a weight stored `{n, k}` alike, filled ahead and through registers.
#[test]
fn the_ldmatrix_transport_reads_every_arrangement() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            for schedule in [Prefetch::InSlots, Prefetch::InRegisters] {
                check(Case {
                    chunks,
                    transposed,
                    depth: 2,
                    schedule,
                    leaf: Leaf::Mma,
                    ..Case::DEFAULT
                });
            }
        }
    }
}

/// An lhs stored `{k, m}`, whose rows `ldmatrix` hands out transposed, beside either rhs, in every
/// arrangement: the manual transport reads the same product.
#[test]
fn the_ldmatrix_transport_reads_a_col_major_lhs() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                transposed,
                lhs_transposed: true,
                ..Case::DEFAULT
            });
        }
    }
}

/// Several blocks along `M` and `N`, rows one instruction deep, and lines narrower than a chunk:
/// the row a unit addresses is still its own, whatever its block and however many lines it holds.
#[test]
fn the_ldmatrix_transport_reads_blocks_shallow_rows_and_narrow_lines() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                mn: 32,
                transposed,
                leaf: Leaf::Mma,
                ..Case::DEFAULT
            });
            check(Case {
                chunks,
                stage_k: 16,
                transposed,
                leaf: Leaf::Mma,
                ..Case::DEFAULT
            });
            check(Case {
                chunks,
                v: 2,
                transposed,
                leaf: Leaf::Mma,
                ..Case::DEFAULT
            });
        }
    }
}

/// Lines wider than an `ldmatrix` row: the transport reads them manually, and the product holds.
#[test]
fn the_ldmatrix_transport_reads_wide_lines_manually() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            v: 16,
            stage_k: 64,
            transposed: true,
            leaf: Leaf::Mma,
            ..Case::DEFAULT
        });
    }
}

/// A stage behind a 4-byte shared allocation still starts its rows on 16 bytes, which an
/// `ldmatrix` row address needs: the stage is aligned
/// ([`RowChunks::STAGE_ALIGNMENT_BYTES`]) whatever is declared before it.
/// In-order lines of every width a row holds whole.
#[test]
fn the_ldmatrix_transport_reads_a_stage_declared_after_another_allocation() {
    for v in [2, 4, 8] {
        for transposed in [false, true] {
            check(Case {
                v,
                transposed,
                staging: Staging::StageAfterSmallAllocation,
                leaf: Leaf::Mma,
                ..Case::DEFAULT
            });
        }
    }
}

/// Fragments read straight out of global memory, which `ldmatrix` cannot address: the transport
/// reads them manually whatever it was asked, a row-major rhs and a weight stored `{n, k}` alike.
#[test]
fn the_ldmatrix_transport_reads_a_global_window_manually() {
    for transposed in [false, true] {
        check(Case {
            transposed,
            staging: Staging::Global,
            leaf: Leaf::Mma,
            ..Case::DEFAULT
        });
    }
}

/// `bf16` operands through `ldmatrix`, arranged every way a unit can address.
#[test]
fn the_ldmatrix_transport_reads_bf16() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                transposed,
                input: ElemType::Float(cubecl::ir::FloatKind::BF16),
                leaf: Leaf::Mma,
                ..Case::DEFAULT
            });
        }
    }
}
