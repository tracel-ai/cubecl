//! A stage whose block rows are swizzled ([`RowChunks::Swizzled`]) or padded
//! ([`RowChunks::Padded`]) holds the same product as one whose rows lie in order: filled whole,
//! filled through registers, and read by the manual mma transport a cell at a time or through
//! `ldmatrix` — or, padded, by the cmma one — a row-major rhs and a weight stored `{n, k}` alike.
//! The `ldmatrix` transport is also held to what it cannot read: a global window, read manually.
//!
//! In `f16` (and `bf16` through `ldmatrix`) summed in `f32`, at `16×16×16` where the device offers
//! it and at the shape it does otherwise (NVIDIA's manual mma is `16×8×16`); a device with no such
//! instruction, or no `ldmatrix` for the element, reports the test skipped. Lines of
//! other byte widths (a 4-byte element's, a packed word's) are arranged by the same rule, held on
//! the host by `RowArrangement`'s and `ChunkSwizzle`'s own tests.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{ir::ElemType, prelude::*, zspace::shape};
use cubecl_tile::ops::matmul::{LoadMethod, MmaIo};
use cubecl_tile::stage::RowChunks;
use cubecl_tile::*;

use super::matmul::Schedule;
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

/// Which transport reads a stage's fragments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Leaf {
    /// The manual mma, each unit computing its own addresses.
    Mma,
    /// The vendor's fragment API, rows off a pointer and a stride.
    Cmma,
    /// The manual mma, both operands' fragments handed out by `ldmatrix` from each unit's
    /// row address: the swizzle applied where the unit addresses its row.
    MmaLoadMatrix,
}

/// `c = a · b` over a walk along `K` whose every region is one stage of `outer`'s tile, read a
/// fragment of `inner` at a time through `leaf`, the stage laid out as `storage`.
#[cube(launch)]
fn staged_k_walk<EA: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Schedule,
    #[comptime] leaf: Leaf,
    #[comptime] staging: Staging,
    #[define(EA)] _sum: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    let mut acc = match comptime!(leaf) {
        Leaf::Mma => c.accumulator::<EA>(
            &a,
            &b,
            comptime!(Instruction::Mma {
                io: MmaIo::manual(),
                operands: MmaOperands::Plain,
            }),
            Semiring::SUM_PROD,
        ),
        Leaf::MmaLoadMatrix => c.accumulator::<EA>(
            &a,
            &b,
            comptime!(Instruction::Mma {
                io: MmaIo {
                    lhs_load_method: LoadMethod::LoadMatrix,
                    rhs_load_method: LoadMethod::LoadMatrix,
                    ..MmaIo::manual()
                },
                operands: MmaOperands::Plain,
            }),
            Semiring::SUM_PROD,
        ),
        Leaf::Cmma => c.accumulator::<EA>(&a, &b, Instruction::Cmma, Semiring::SUM_PROD),
    };
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
            comptime!(inner.clone()),
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
#[allow(clippy::too_many_arguments)]
fn staged(
    a: &Tile,
    b: &Tile,
    acc: &mut Tile,
    walk: Walk,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Schedule,
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
#[allow(clippy::too_many_arguments)]
fn walk_stages(
    a: &Tile,
    b: &Tile,
    acc: &mut Tile,
    walk: Walk,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Schedule,
) {
    let mut stages = Stages::smem(&walk, a, b, storage, depth, None);
    match comptime!(schedule) {
        Schedule::AheadInSlots => {
            stages.pipelined(walk, |slot, region| {
                let acc_o = acc.at(region);
                slot.consume(|a_s, b_s| {
                    for fragment in region.over(&inner).unrolled() {
                        let mut acc_f = acc_o.at(&fragment);
                        acc_f.mma(&a_s.at(&fragment), &b_s.at(&fragment));
                    }
                });
            });
        }
        Schedule::ThroughRegisters => {
            stages.prefetched(walk, |slot, region| {
                let acc_o = acc.at(region);
                slot.consume(|a_s, b_s| {
                    for fragment in region.over(&inner).unrolled() {
                        let mut acc_f = acc_o.at(&fragment);
                        acc_f.mma(&a_s.at(&fragment), &b_s.at(&fragment));
                    }
                });
            });
        }
    }
}

/// The `m × n × k` shape this device offers `leaf` at in `input` summed in `f32`: `16×16×16` where
/// it does, else the shallowest other at least `line_depth` deep, else the shallowest, so the
/// stages are read by the instruction each device has. A case whose lines run along `K` asks for a
/// line's depth: a step must start on a line, and NVIDIA lists its `16×8×8` mma before its
/// `16×8×16` one.
/// Where `reads_ldmatrix`, the device must also load `input` through `ldmatrix`; a case the
/// transport reads manually whatever it was asked needs only the mma. Reported rather than silently
/// passed where the device has neither.
fn instruction_shape(
    client: &cubecl::client::Client,
    leaf: Leaf,
    input: ElemType,
    reads_ldmatrix: bool,
    line_depth: usize,
) -> Option<(usize, usize, usize)> {
    let f32 = f32::elem_type_native();
    let matmul = &client.properties().features.matmul;
    let configs = match leaf {
        Leaf::Mma | Leaf::MmaLoadMatrix => &matmul.mma,
        Leaf::Cmma => &matmul.cmma,
    };
    if reads_ldmatrix && !matmul.ldmatrix.contains(&input) {
        ValidationResult::Skipped(format!("device has no {input:?} ldmatrix")).enforce();
        return None;
    }
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
    schedule: Schedule,
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
        schedule: Schedule::AheadInSlots,
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
    // Where the transport reads through `ldmatrix`: a stage whose lines a 16-byte row holds whole.
    // A global window or wider lines are read manually whatever the case asked.
    let reads_ldmatrix = leaf == Leaf::MmaLoadMatrix
        && staging != Staging::Global
        && (16 / input.size()).is_multiple_of(v);
    // A `{m, k}` lhs or a `{n, k}` rhs runs its lines along `K`.
    let lines_along_k = !lhs_transposed || transposed;
    let line_depth = if lines_along_k { v } else { 1 };
    let Some((edge_m, edge_n, edge_k)) =
        instruction_shape(&client, leaf, input, reads_ldmatrix, line_depth)
    else {
        return;
    };
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
        leaf,
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
        for schedule in [Schedule::AheadInSlots, Schedule::ThroughRegisters] {
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
            for schedule in [Schedule::AheadInSlots, Schedule::ThroughRegisters] {
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
            for schedule in [Schedule::AheadInSlots, Schedule::ThroughRegisters] {
                check(Case {
                    chunks,
                    transposed,
                    depth: 2,
                    schedule,
                    leaf: Leaf::MmaLoadMatrix,
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
    for leaf in [Leaf::Mma, Leaf::MmaLoadMatrix] {
        for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
            for transposed in [false, true] {
                check(Case {
                    chunks,
                    transposed,
                    lhs_transposed: true,
                    leaf,
                    ..Case::DEFAULT
                });
            }
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
                leaf: Leaf::MmaLoadMatrix,
                ..Case::DEFAULT
            });
            check(Case {
                chunks,
                stage_k: 16,
                transposed,
                leaf: Leaf::MmaLoadMatrix,
                ..Case::DEFAULT
            });
            check(Case {
                chunks,
                v: 2,
                transposed,
                leaf: Leaf::MmaLoadMatrix,
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
            leaf: Leaf::MmaLoadMatrix,
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
                leaf: Leaf::MmaLoadMatrix,
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
            leaf: Leaf::MmaLoadMatrix,
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
                leaf: Leaf::MmaLoadMatrix,
                ..Case::DEFAULT
            });
        }
    }
}
