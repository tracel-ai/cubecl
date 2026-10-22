//! A paged KV cache, which is expert routing with the axis split.
//!
//! The physical pool is `[PAGE, OFFSET]` over one buffer, spelled with
//! [`PhysicalAxisMap::disjoint`] exactly as a quantization block splits `K`. A page table names
//! which physical page holds a sequence's logical one, and the walk routes `PAGE` to it.
//!
//! So the pool needs no ordering of its own, and the page size is the split's radix rather than a
//! number carried beside a lookup.
//!
//! The second half reads the same kind of pool with no walk over pages at all: the operand is
//! bound [`paged`](cubecl_tile::launch::Arg::paged), the kernel walks logical positions, and the
//! table is resolved where each line's offset is taken.
#![allow(non_snake_case)]

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::Shape};
use cubecl_tile::launch::{Arg, Delivery, Labelled, Refusal};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;

/// Which sequence.
const B: Axis = Axis(0);
/// Which of the sequence's own pages: what the table is addressed by.
const LP: Axis = Axis(1);
/// Which physical page of the pool: what the table names, and what the walk routes.
const PAGE: Axis = Axis(2);
/// Where inside a page.
const OFFSET: Axis = Axis(3);
/// The feature a slot carries.
const D: Axis = Axis(4);

const SEQS: usize = 2;
const PAGES_PER_SEQ: usize = 2;
const POOL_PAGES: usize = 4;
const PAGE_SIZE: usize = 2;
const FEATURES: usize = 2;
const SLOTS: usize = POOL_PAGES * PAGE_SIZE;

/// Page 2 is shared: sequence 0 holds it first, sequence 1 second. Nothing about the pool's order
/// says so, which is the point of a table.
const TABLE: [u32; SEQS * PAGES_PER_SEQ] = [2, 0, 3, 2];

/// `out[b] = Σ kv[slot]` over the slots of the pages sequence `b` holds.
#[cube(launch)]
fn paged_sum_kernel<E: Numeric>(
    kv: &TileArg<'_>,
    out: &TileArg<'_>,
    table: &Tensor<u32>,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] logical: Level,
    #[comptime] physical: Level,
    #[define(E)] _dtype: ElemType,
) {
    let kv = kv.tile(&space);
    let out = out.tile(&space);

    for sequence in space.over(&seq) {
        let b = sequence.coord(B);

        let mut total = out.at(&sequence);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        for lp in sequence.over(&logical) {
            let phys = table[b * PAGES_PER_SEQ + lp.coord(LP)] as usize;

            // The same call MoE makes: the axis keeps the pool's four pages, the walk visits the
            // one the table named.
            for page in lp.over(&physical).routed(PAGE, phys) {
                let mut cell = out.at(&page);
                cell.reduce_axis_accumulate(&kv.at(&page), comptime!(Monoid::Sum));
            }
        }
    }
}

/// Every slot distinct, so reading a neighbouring page cannot land on the right answer.
fn kv_values() -> Vec<f32> {
    (0..SLOTS * FEATURES)
        .map(|i| {
            let (slot, d) = (i / FEATURES, i % FEATURES);
            (slot * 10 + d) as f32
        })
        .collect()
}

fn run(table: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (B, SEQS),
                (LP, PAGES_PER_SEQ),
                (PAGE, POOL_PAGES),
                (OFFSET, PAGE_SIZE),
                (D, FEATURES),
            ]),
            Levels::leaf(&[(B, 1), (LP, 1), (PAGE, 1)])
                .walk_every(&[PAGE])
                .walk_every(&[LP])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (kv_handle, _) = TestInput::builder(client.clone(), Shape::new([SLOTS, FEATURES]))
        .dtype(f32_ty)
        .custom(kv_values())
        .generate_with_f32_host_data();
    let table_handle = TestInput::builder(client.clone(), Shape::new([TABLE.len()]))
        .dtype(u32_ty)
        .custom(table.iter().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQS * FEATURES])
        .generate_without_host_data();

    paged_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            kv_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            // The pool's one dimension read as two axes: `slot = page * PAGE_SIZE + offset`.
            TileSpec::new(Projection::new(
                &[PAGE, OFFSET, D],
                &[
                    PhysicalAxisMap::disjoint(&[(PAGE, PAGE_SIZE), (OFFSET, 1)]),
                    PhysicalAxisMap::of(D),
                ],
            )),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, D]),
        ),
        table_handle.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        launcher.partitioning().level(2),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference: each sequence over the slots of the pages its table names.
fn expected(table: &[u32]) -> Vec<Vec<f32>> {
    let values = kv_values();
    (0..SEQS)
        .map(|b| {
            (0..FEATURES)
                .map(|d| {
                    (0..PAGES_PER_SEQ)
                        .flat_map(|lp| {
                            let page = table[b * PAGES_PER_SEQ + lp] as usize;
                            (0..PAGE_SIZE).map(move |o| (page * PAGE_SIZE + o) * FEATURES + d)
                        })
                        .map(|i| values[i])
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Two page tables over the same pool: the answer follows the table, which it could not if the
/// route did nothing, since the walk would fold the whole pool under both.
#[test]
fn a_routed_page_axis_folds_each_sequence_over_the_pages_it_holds() {
    for table in [TABLE, [0, 1, 1, 0]] {
        let got = run(&table);
        let want = expected(&table);
        for (b, row) in want.iter().enumerate() {
            for (d, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, d]),
                    *cell,
                    "sequence {b} feature {d}: pages {:?}",
                    &table[b * PAGES_PER_SEQ..(b + 1) * PAGES_PER_SEQ]
                );
            }
        }
    }
}

/// The paged reads below: a position, as the sequences see it, and nothing of the pool.
const S: Axis = Axis(1);
/// The feature a position carries.
const F: Axis = Axis(2);

const SEQUENCES: usize = 3;
const PAGES_EACH: usize = 3;
const PAGE_LEN: usize = 2;
const POOL: usize = 7;
const FEATURES_READ: usize = 3;
/// Positions a step reads: more than a page, so every step but the last crosses one.
const READ_BLOCK: usize = 4;

/// Each sequence's pages, scattered over the pool; page 4 is sequence 2's twice, and page 0 is
/// held by two sequences.
const PAGES: [[u32; PAGES_EACH]; SEQUENCES] = [[5, 0, 3], [1, 6, 2], [4, 4, 0]];

/// `out[b] = Σ pool[b's page of s, s % PAGE_LEN]` over positions `from..lens[b]` of sequence `b`,
/// read through [`TileArg::tile_paged`]: the kernel walks logical positions, `read_block` of them
/// a step, and [`Tile::within`] cuts each sequence at its own length.
#[cube(launch)]
fn paged_read_sum_kernel<E: Numeric>(
    pool: &TileArg<'_>,
    pages: ComptimeOption<&Tensor<u32>>,
    lens: &Tensor<u32>,
    out: &TileArg<'_>,
    #[comptime] from: usize,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] block: Level,
    #[comptime] read_block: usize,
    #[define(E)] _dtype: ElemType,
) {
    let pool = pool.tile_paged(&space, pages);
    let out = out.tile(&space);

    for sequence in space.over(&seq) {
        let len = lens[sequence.coord(B)] as usize;

        let mut total = out.at(&sequence);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        let positions = pool.within(S, from, len);
        for blk in sequence
            .over(&block)
            .range(0, len.saturating_sub(from).div_ceil(read_block))
        {
            let mut cell = out.at(&blk);
            cell.reduce_axis_accumulate(&positions.at(&blk), comptime!(Monoid::Sum));
        }
    }
}

/// Every pool cell distinct, pages of `page_len` positions: `page·100 + row·10 + feature`.
fn pool_values(page_len: usize) -> Vec<f32> {
    (0..POOL * page_len * FEATURES_READ)
        .map(|i| {
            let (page, row, f) = (
                i / (page_len * FEATURES_READ),
                (i / FEATURES_READ) % page_len,
                i % FEATURES_READ,
            );
            (page * 100 + row * 10 + f) as f32
        })
        .collect()
}

fn run_paged_read(lens: &[u32; SEQUENCES]) -> HostData {
    run_paged_read_over(lens, &rows(&PAGES))
}

/// A read through [`PAGES`] `block` positions a step from position `from` on.
fn run_paged_read_from(lens: &[u32; SEQUENCES], block: usize, from: u32) -> HostData {
    run_paged_read_blocks(lens, &rows(&PAGES), block, from)
}

/// `pages` as a table of rows, any number of them and of any width.
fn rows(pages: &[[u32; PAGES_EACH]]) -> Vec<Vec<u32>> {
    pages.iter().map(|row| row.to_vec()).collect()
}

/// [`run_paged_read`] through `pages`, whose rows and columns need not be the launch's: the table
/// sets the operand's extent, and the launch walks its own.
fn run_paged_read_over(lens: &[u32; SEQUENCES], pages: &[Vec<u32>]) -> HostData {
    run_paged_read_blocks(lens, pages, READ_BLOCK, 0)
}

/// [`run_paged_read_over`] `block` positions a step from position `from` on.
fn run_paged_read_blocks(
    lens: &[u32; SEQUENCES],
    pages: &[Vec<u32>],
    block: usize,
    from: u32,
) -> HostData {
    let columns = pages.first().map_or(PAGES_EACH, Vec::len);
    run_paged_read_table(
        lens,
        &[pages.len(), columns],
        pages.iter().flatten().copied().collect(),
        PAGE_LEN,
        block,
        from,
        |arg, s, p| arg.paged(s, p),
    )
}

/// How a read binds its pool: [`Arg::paged`] or [`Arg::paged_covering`].
type Bind = for<'a> fn(Arg<'a, Labelled>, Axis, Axis) -> Arg<'a, Labelled>;

/// [`run_paged_read_blocks`] through a table of `entries` laid out as `shape`, any rank, the pool
/// bound by `bind`.
fn run_paged_read_table(
    lens: &[u32; SEQUENCES],
    shape: &[usize],
    entries: Vec<u32>,
    page_len: usize,
    block: usize,
    from: u32,
    bind: Bind,
) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (B, SEQUENCES),
                (S, PAGES_EACH * page_len),
                (F, FEATURES_READ),
            ]),
            Levels::leaf(&[(B, 1), (S, block)])
                .walk_every(&[S])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (pool_handle, _) =
        TestInput::builder(client.clone(), Shape::new([POOL, page_len, FEATURES_READ]))
            .dtype(f32_ty)
            .custom(pool_values(page_len))
            .generate_with_f32_host_data();
    let table_handle = TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(u32_ty)
        .custom(entries.iter().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let lens_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES]))
        .dtype(u32_ty)
        .custom(lens.iter().map(|&l| l as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES, FEATURES_READ]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQUENCES * FEATURES_READ])
        .generate_without_host_data();

    let pool = bind(
        launcher.arg(pool_handle.binding(), f32_ty).axes(&[B, S, F]),
        B,
        S,
    )
    .build()
    .unwrap();

    paged_read_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        pool.arg(),
        ComptimeOptionArgs::Some(table_handle.binding().into_tensor_arg()),
        lens_handle.binding().into_tensor_arg(),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, F]),
        ),
        from as usize,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        block,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// [`paged_read_sum_kernel`] walking each sequence a page at a time and the page's positions
/// `step` at a time inside it: each page's window is held and its steps cut from it. A `from` of
/// 0 keeps the alignment the levels prove; any other is a runtime shift that loses it.
#[cube(launch)]
fn paged_page_walk_sum_kernel<E: Numeric>(
    pool: &TileArg<'_>,
    pages: ComptimeOption<&Tensor<u32>>,
    lens: &Tensor<u32>,
    out: &TileArg<'_>,
    #[comptime] from: usize,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] page: Level,
    #[comptime] page_len: usize,
    #[define(E)] _dtype: ElemType,
) {
    let pool = pool.tile_paged(&space, pages);
    let out = out.tile(&space);

    for sequence in space.over(&seq) {
        let len = lens[sequence.coord(B)] as usize;

        let mut total = out.at(&sequence);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        let positions = pool.within(S, from, len);
        for held in sequence
            .over(&page)
            .range(0, len.saturating_sub(from).div_ceil(page_len))
        {
            let page_window = positions.at(&held);
            for step in held.walk() {
                let mut cell = out.at(&step);
                cell.reduce_axis_accumulate(&page_window.at(&step), comptime!(Monoid::Sum));
            }
        }
    }
}

/// Positions a page holds in the page walk: several steps of more than one position fit in it.
const WALK_PAGE_LEN: usize = 4;

/// [`paged_page_walk_sum_kernel`] over [`PAGES`] of [`WALK_PAGE_LEN`] positions, `step` positions
/// a step inside each page, from position `from` on.
fn run_paged_page_walk(lens: &[u32; SEQUENCES], step: usize, from: u32) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (B, SEQUENCES),
                (S, PAGES_EACH * WALK_PAGE_LEN),
                (F, FEATURES_READ),
            ]),
            Levels::leaf(&[(B, 1), (S, step)])
                .walk(&[(S, WALK_PAGE_LEN / step)])
                .walk_every(&[S])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (pool_handle, _) = TestInput::builder(
        client.clone(),
        Shape::new([POOL, WALK_PAGE_LEN, FEATURES_READ]),
    )
    .dtype(f32_ty)
    .custom(pool_values(WALK_PAGE_LEN))
    .generate_with_f32_host_data();
    let table_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES, PAGES_EACH]))
        .dtype(u32_ty)
        .custom(PAGES.iter().flatten().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let lens_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES]))
        .dtype(u32_ty)
        .custom(lens.iter().map(|&l| l as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES, FEATURES_READ]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQUENCES * FEATURES_READ])
        .generate_without_host_data();

    let pool = launcher
        .arg(pool_handle.binding(), f32_ty)
        .axes(&[B, S, F])
        .paged_covering(B, S)
        .build()
        .unwrap();

    paged_page_walk_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        pool.arg(),
        ComptimeOptionArgs::Some(table_handle.binding().into_tensor_arg()),
        lens_handle.binding().into_tensor_arg(),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, F]),
        ),
        from as usize,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        WALK_PAGE_LEN,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference: each sequence's positions looked up in its row of the table on the host.
fn expected_paged_read(lens: &[u32; SEQUENCES]) -> Vec<Vec<f32>> {
    expected_paged_read_over(lens, &rows(&PAGES))
}

/// [`expected_paged_read`] through `pages`, an entry past the pool reading its last page and a
/// position past the table reading zero.
fn expected_paged_read_over(lens: &[u32; SEQUENCES], pages: &[Vec<u32>]) -> Vec<Vec<f32>> {
    expected_paged_read_from(lens, pages, 0, PAGE_LEN)
}

/// [`expected_paged_read_over`] from position `from` on, over pages of `page_len` positions.
fn expected_paged_read_from(
    lens: &[u32; SEQUENCES],
    pages: &[Vec<u32>],
    from: u32,
    page_len: usize,
) -> Vec<Vec<f32>> {
    let values = pool_values(page_len);
    (0..SEQUENCES)
        .map(|b| {
            (0..FEATURES_READ)
                .map(|f| {
                    (from as usize..lens[b] as usize)
                        .map(|s| {
                            let page = pages.get(b).and_then(|row| row.get(s / page_len));
                            page.map_or(0.0, |&page| {
                                let page = (page as usize).min(POOL - 1);
                                values[(page * page_len + s % page_len) * FEATURES_READ + f]
                            })
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// A sequence read through its page table folds the positions its pages hold, in its table's
/// order: steps that cross pages, a length that ends inside a page, a full sequence and one of a
/// single position. Reading the pool where it lies would fold pages 0, 1 and 2 for sequence 0.
#[test]
fn a_paged_operand_reads_each_sequence_through_its_table() {
    for lens in [[5, 6, 1], [6, 3, 4]] {
        let got = run_paged_read(&lens);
        let want = expected_paged_read(&lens);
        for (b, row) in want.iter().enumerate() {
            for (f, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, f]),
                    *cell,
                    "sequence {b} feature {f}: {} positions of pages {:?}",
                    lens[b],
                    PAGES[b]
                );
            }
        }
    }
}

/// A table entry past the pool reads its last page rather than past the buffer: a table is device
/// data a launch may never have written.
#[test]
fn a_paged_read_stays_in_the_pool_and_the_table() {
    let pages = rows(&[[5, 99, 3], [1, 6, 2], [4, u32::MAX, 0]]);
    let lens = [6, 3, 5];
    let got = run_paged_read_over(&lens, &pages);
    let want = expected_paged_read_over(&lens, &pages);
    for (b, row) in want.iter().enumerate() {
        for (f, cell) in row.iter().enumerate() {
            assert_eq!(got.get_f32(&[b, f]), *cell, "sequence {b} feature {f}");
        }
    }
}

/// Steps of a page or less resolve both pages a window can touch as it is cut: aligned to the
/// pages, inside one, and straddling two from an odd start.
#[test]
fn a_paged_read_of_steps_within_a_page_reads_each_sequence_through_its_table() {
    let lens = [6, 5, 3];
    for (block, from) in [(1, 0), (2, 0), (2, 1), (1, 3)] {
        let got = run_paged_read_from(&lens, block, from);
        let want = expected_paged_read_from(&lens, &rows(&PAGES), from, PAGE_LEN);
        for (b, row) in want.iter().enumerate() {
            for (f, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, f]),
                    *cell,
                    "{block} a step from {from}: sequence {b} feature {f}"
                );
            }
        }
    }
}

/// Steps cut straight from the whole sequence, over pages of four positions: a step of 1, 2 or 4
/// has an origin aligned to its own size, which proves it inside one page; a step of 3 proves
/// nothing and may straddle two.
#[test]
fn a_flat_walk_over_larger_pages_reads_each_sequence_through_its_table() {
    let entries = PAGES.iter().flatten().copied().collect::<Vec<_>>();
    let lens = [9, 12, 7];
    for (block, from) in [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (4, 3)] {
        let got = run_paged_read_table(
            &lens,
            &[SEQUENCES, PAGES_EACH],
            entries.clone(),
            WALK_PAGE_LEN,
            block,
            from,
            |arg, s, p| arg.paged(s, p),
        );
        let want = expected_paged_read_from(&lens, &rows(&PAGES), from, WALK_PAGE_LEN);
        for (b, row) in want.iter().enumerate() {
            for (f, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, f]),
                    *cell,
                    "steps of {block} from {from}: sequence {b} feature {f}"
                );
            }
        }
    }
}

/// A sequence walked a page at a time reads each page through the table once and every step
/// inside it through that page: steps of one position, of two and of the whole page, lengths
/// ending inside a page, over pages scattered and shared across sequences. Started past 0, each
/// page's window straddles two pages, so its steps cannot keep its pin.
#[test]
fn a_sequence_walked_a_page_at_a_time_reads_each_sequence_through_its_table() {
    for (step, from) in [
        (1, 0),
        (2, 0),
        (WALK_PAGE_LEN, 0),
        (2, 1),
        (1, 3),
        (WALK_PAGE_LEN, 3),
    ] {
        for lens in [[9, 12, 1], [12, 6, 7]] {
            let got = run_paged_page_walk(&lens, step, from);
            let want = expected_paged_read_from(&lens, &rows(&PAGES), from, WALK_PAGE_LEN);
            for (b, row) in want.iter().enumerate() {
                for (f, cell) in row.iter().enumerate() {
                    assert_eq!(
                        got.get_f32(&[b, f]),
                        *cell,
                        "steps of {step} from {from}: sequence {b} feature {f}, {} positions",
                        lens[b]
                    );
                }
            }
        }
    }
}

/// The table sets the operand's extent, not the launch: a table of fewer rows or narrower rows
/// than the launch walks reads zero past them, down to a table of one entry. That no load lands
/// past the table is `PageTable::lookup`'s clamp, which values cannot show: the mask drops
/// whatever such a load returns.
#[test]
fn a_paged_read_past_the_table_reads_zero() {
    let lens = [6, 5, 4];
    for pages in [
        rows(&PAGES[..2]),
        PAGES.iter().map(|row| row[..2].to_vec()).collect(),
        vec![vec![PAGES[0][0]]],
    ] {
        let got = run_paged_read_over(&lens, &pages);
        let want = expected_paged_read_over(&lens, &pages);
        for (b, row) in want.iter().enumerate() {
            for (f, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, f]),
                    *cell,
                    "sequence {b} feature {f} through {pages:?}"
                );
            }
        }
    }
}

/// A table stated to cover the launch reads as one checked against its shape wherever the
/// launch keeps its reads in bounds: steps that cross pages, lengths ending inside a page.
#[test]
fn a_covering_table_reads_each_sequence_through_it() {
    let entries = PAGES.iter().flatten().copied().collect::<Vec<_>>();
    for lens in [[5, 6, 1], [6, 3, 4]] {
        let shape = [SEQUENCES, PAGES_EACH];
        let got = run_paged_read_table(
            &lens,
            &shape,
            entries.clone(),
            PAGE_LEN,
            READ_BLOCK,
            0,
            |arg, s, p| arg.paged_covering(s, p),
        );
        for (b, row) in expected_paged_read(&lens).iter().enumerate() {
            for (f, cell) in row.iter().enumerate() {
                assert_eq!(got.get_f32(&[b, f]), *cell, "sequence {b} feature {f}");
            }
        }
    }
}

/// A table is `[sequences, pages_per_sequence]`: one of any other rank, here the same entries
/// flat, names no page for any sequence and reads zero throughout, never its missing dims.
#[test]
fn a_paged_read_through_a_table_not_of_rank_two_reads_zero() {
    let lens = [6, 5, 4];
    let entries: Vec<u32> = PAGES.iter().flatten().copied().collect();
    let got = run_paged_read_table(
        &lens,
        &[entries.len()],
        entries,
        PAGE_LEN,
        READ_BLOCK,
        0,
        |arg, s, p| arg.paged(s, p),
    );
    for b in 0..SEQUENCES {
        for f in 0..FEATURES_READ {
            assert_eq!(got.get_f32(&[b, f]), 0.0, "sequence {b} feature {f}");
        }
    }
}

/// What a kernel could not read through a table is refused as the operand is bound: one dim for
/// both axes, pages along the innermost dim, a line that would cross a page (stride order
/// included, which moves the positions innermost), a pool of no pages, and a delivery that moves
/// runs.
#[test]
fn a_paged_operand_the_table_cannot_resolve_is_refused_at_build() {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, SEQUENCES), (S, PAGES_EACH * 3), (F, 4)]),
            Levels::leaf(&[(B, 1), (S, READ_BLOCK)])
                .walk_every(&[S])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );
    let pool = |shape: [usize; 3]| {
        TestInput::builder(client.clone(), Shape::new(shape))
            .dtype(f32::elem_type_native())
            .zeros()
            .generate_without_host_data()
    };
    let refusal = |binding, axes: &[Axis], (sequence, position), width| {
        launcher
            .arg(binding, f32::elem_type_native())
            .axes(axes)
            .vectorize(width)
            .paged(sequence, position)
            .build()
            .err()
    };
    let a = pool([POOL, 3, 4]);
    assert_eq!(
        refusal(a.binding(), &[B, S, F], (S, S), 1),
        Some(Refusal::PagingOfOneDim(S))
    );
    let b = pool([POOL, 3, 4]);
    assert_eq!(
        refusal(b.binding(), &[S, F, B], (B, S), 1),
        Some(Refusal::PagesAlongALine(B))
    );
    let c = pool([POOL, 4, 3]);
    assert_eq!(
        refusal(c.binding(), &[B, F, S], (B, S), 2),
        Some(Refusal::PageSplitsALine {
            page_size: 3,
            width: 2
        })
    );
    // The same pool stated `[B, S, F]` over its strides: only stride order puts S innermost.
    let d = pool([POOL, 4, 3]);
    let mut strided = d.binding();
    strided.shape = [POOL, 3, 4].into();
    strided.strides = [12, 1, 3].into();
    let reordered = launcher
        .arg(strided, f32::elem_type_native())
        .axes(&[B, S, F])
        .in_stride_order()
        .vectorize(2)
        .paged(B, S)
        .build()
        .err();
    assert_eq!(
        reordered,
        Some(Refusal::PageSplitsALine {
            page_size: 3,
            width: 2
        })
    );
    let e = pool([0, 3, 4]);
    assert_eq!(
        refusal(e.binding(), &[B, S, F], (B, S), 1),
        Some(Refusal::PagingOfAnEmptyPool)
    );
    let f = pool([POOL, 3, 4]);
    let by_runs = launcher
        .arg(f.binding(), f32::elem_type_native())
        .axes(&[B, S, F])
        .delivery(Delivery::Tma)
        .paged(B, S)
        .build()
        .err();
    assert_eq!(by_runs, Some(Refusal::PagingMovedByRuns));
}

/// The pool is bound as it lies, so where its pages sit among its dims is no matter: pages inner
/// to a head dim, and a head dim of one whose stride is anything, as an `unsqueeze` leaves it.
#[test]
fn a_paged_pool_binds_wherever_its_pages_lie() {
    const H: Axis = Axis(3);
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, SEQUENCES), (S, PAGES_EACH * PAGE_LEN), (F, 4), (H, 2)]),
            Levels::leaf(&[(B, 1), (S, READ_BLOCK)])
                .walk_every(&[S])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );
    let pool = TestInput::builder(client.clone(), Shape::new([2, POOL, PAGE_LEN, 4]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let heads_outside = launcher
        .arg(pool.binding(), f32::elem_type_native())
        .axes(&[H, B, S, F])
        .paged(B, S)
        .build();
    assert!(heads_outside.is_ok(), "{:?}", heads_outside.err());
    let unsqueezed = TestInput::builder(client.clone(), Shape::new([POOL, PAGE_LEN, 4]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    let mut binding = unsqueezed.binding();
    binding.shape = [POOL, 1, PAGE_LEN, 4].into();
    binding.strides = [PAGE_LEN * 4, 4096, 4, 1].into();
    let one_head = launcher
        .arg(binding, f32::elem_type_native())
        .axes(&[B, H, S, F])
        .paged(B, S)
        .build();
    assert!(one_head.is_ok(), "{:?}", one_head.err());
}
