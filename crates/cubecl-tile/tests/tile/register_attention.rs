//! A flash-attention leaf whose score never leaves registers: `q · kᵀ` lands in manual-mma
//! accumulator fragments, the online softmax runs on them with each row reduced across the four
//! units that hold it, and the accumulator, packed to `f16`, is the `A` fragment of `p · v`.
//!
//! The identity that makes the last step free: an `m16n8` accumulator's two `n8` tiles side by
//! side hold, cell for cell, the registers of an `m16k16` `A` fragment. The component test below
//! checks it through [`MmaDefinition::position_of_nth`] rather than assuming it.
//!
//! Each cube is four planes of sixteen query rows of one head; keys and values are staged in
//! shared memory a block of [`KEYS`] at a time, two stages deep so the next block lands by async
//! copy while this one computes, and read by `ldmatrix`, the values transposed on the way. The
//! correctness check runs against a host reference; the throughput is a measurement, ignored by
//! default:
//!
//! ```sh
//! cargo test -p cubecl-tile --release --features cubecl/cuda,cubecl/cuda-cpp --test lib \
//!     -- register_attention --ignored --nocapture
//! ```

use std::time::Instant;

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    prelude::barrier::{Barrier, copy_async},
    prelude::*,
};
use cubecl_tile::Monoid;
use half::f16;

/// The head dim, which is also the value dim.
const DIM: usize = 128;

/// Query rows one plane owns: the instruction's `m`.
const ROWS: usize = 16;

/// Planes per cube.
const PLANES: usize = 4;

/// Keys one stage holds.
const KEYS: usize = 64;

/// Cells a 16-byte line of `f16` holds.
const LINE: usize = 8;

/// Lines a staged row spans, one more than the dim's: the padding line shifts each row four banks,
/// so the eight rows one `ldmatrix` reads land in distinct banks.
const ROW_LINES: usize = DIM / LINE + 1;

/// `out = softmax(q · kᵀ · scale) · v` per head, `group` query heads to a key-value head, the
/// score's rows reduced in registers. `q` and `out` are `[heads, queries, DIM]`, `k` and `v`
/// `[heads / group, keys, DIM]`, all contiguous.
#[cube(launch)]
fn register_attention(
    q: &Tensor<f16>,
    k: &Tensor<Vector<f16, Const<8>>>,
    v: &Tensor<Vector<f16, Const<8>>>,
    out: &mut Tensor<f16>,
    queries: u32,
    keys: u32,
    scale: f32,
    #[comptime] group: u32,
    #[comptime] value_depth: usize,
) {
    let def = MmaDefinition::<f16, f16, f32>::new(ROWS, 8usize, 16usize);
    let size!(NA) = def.vector_size(MatrixIdent::A);
    let size!(NB) = def.vector_size(MatrixIdent::B);
    let size!(NC) = def.vector_size(MatrixIdent::Accumulator);
    let a_regs = def.vectors_per_lane(MatrixIdent::A);
    let b_regs = def.vectors_per_lane(MatrixIdent::B);
    let c_regs = def.vectors_per_lane(MatrixIdent::Accumulator);
    let a_size = def.vector_size(MatrixIdent::A);
    let c_size = def.vector_size(MatrixIdent::Accumulator);
    let b_layout = def.vector_layout(MatrixIdent::B);
    // Keys are staged `[key, dim]`: `kᵀ`'s `B` lies col-major in them, `v`'s row-major.
    let key_transposed = comptime!(b_layout != MatrixLayout::ColMajor);
    let value_transposed = comptime!(b_layout != MatrixLayout::RowMajor);

    let lane = UNIT_POS_PLANE;
    let plane = UNIT_POS_Y;
    let head = CUBE_POS_Y;
    let kv_head = head / group;
    let first_row = CUBE_POS_X * comptime!((ROWS * PLANES) as u32) + plane * comptime!(ROWS as u32);

    // The plane's query rows as `A` fragments, one per 16-wide step of the head dim.
    let depth_steps = comptime!(DIM / 16);
    let mut query = Array::<Vector<f16, NA>>::new(comptime!(depth_steps * a_regs));
    let q_base = (head * queries + first_row) * comptime!(DIM as u32);
    #[unroll]
    for step in 0..depth_steps {
        #[unroll]
        for i in 0..a_regs {
            let mut reg = Vector::empty();
            #[unroll]
            for e in 0..a_size {
                let (row, col) =
                    def.position_of_nth(lane, comptime!(i * a_size + e) as u32, MatrixIdent::A);
                let at = q_base + row * comptime!(DIM as u32) + comptime!(step as u32 * 16) + col;
                reg.insert(e, q[at as usize]);
            }
            query[comptime!(step * a_regs + i)] = reg;
        }
    }

    // `KEYS / 8` score tiles of `m16n8`, and `DIM / 8` output tiles.
    let score_tiles = comptime!(KEYS / 8);
    let out_tiles = comptime!(DIM / 8);
    let mut score = Array::<Vector<f32, NC>>::new(comptime!(score_tiles * c_regs));
    let mut acc = Array::<Vector<f32, NC>>::new(comptime!(out_tiles * c_regs));
    #[unroll]
    for i in 0..comptime!(out_tiles * c_regs) {
        acc[i] = Vector::cast_from(0.0f32);
    }
    // One running max (of scaled scores) and one partial sum per accumulator register row: the
    // unit's own cells only, its quad's merged at the end.
    let mut running_max = Array::<f32>::new(c_regs);
    let mut sum = Array::<f32>::new(c_regs);
    #[unroll]
    for r in 0..c_regs {
        running_max[r] = -1.0e30f32;
        sum[r] = 0.0f32;
    }

    // Two stages: the next block's keys and values land by async copy while this one computes.
    let stage_lines = comptime!(KEYS * ROW_LINES);
    let mut key_stage =
        Shared::<[Vector<f16, Const<8>>]>::new_aligned_slice(comptime!(2 * stage_lines), 16usize);
    let mut value_stage =
        Shared::<[Vector<f16, Const<8>>]>::new_aligned_slice(comptime!(2 * stage_lines), 16usize);
    let landed_even = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    let landed_odd = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    let kv_base = kv_head * keys * comptime!((DIM / LINE) as u32);
    let row_in_matrix = lane % 8;
    let nth_matrix = lane / 8 % comptime!(b_regs as u32);
    let b_size = def.vector_size(MatrixIdent::B);
    let (b_row, b_col) =
        def.position_of_nth(0, nth_matrix * comptime!(b_size as u32), MatrixIdent::B);

    let blocks = keys / comptime!(KEYS as u32);
    let block_lines = comptime!((KEYS * DIM / LINE) as u32);
    fill_stage(k, v, &mut key_stage, &mut value_stage, kv_base, 0);
    landed_even.commit_copy_async();
    for block in 0..blocks {
        let stage = block % 2;
        let stage_base = stage * comptime!(stage_lines as u32);
        // The other stage was read by the previous block: every plane is past it before it refills.
        sync_cube();
        if block + 1 < blocks {
            let next_base = kv_base + (block + 1) * block_lines;
            fill_stage(
                k,
                v,
                &mut key_stage,
                &mut value_stage,
                next_base,
                (1 - stage) * comptime!(stage_lines as u32),
            );
            if stage == 0 {
                landed_odd.commit_copy_async();
            } else {
                landed_even.commit_copy_async();
            }
        }
        if stage == 0 {
            landed_even.arrive_and_wait();
        } else {
            landed_odd.arrive_and_wait();
        }

        // score = q · kᵀ, the keys the columns.
        #[unroll]
        for i in 0..comptime!(score_tiles * c_regs) {
            score[i] = Vector::cast_from(0.0f32);
        }
        #[unroll]
        for tile in 0..score_tiles {
            let mut c = Array::<Vector<f32, NC>>::new(c_regs);
            #[unroll]
            for r in 0..c_regs {
                c[r] = score[comptime!(tile * c_regs + r)];
            }
            #[unroll]
            for step in 0..depth_steps {
                let mut a = Array::<Vector<f16, NA>>::new(a_regs);
                #[unroll]
                for r in 0..a_regs {
                    a[r] = query[comptime!(step * a_regs + r)];
                }
                // `kᵀ`'s `(dim, key)` lies at the stage's `(key, dim)`.
                let key = comptime!(tile as u32 * 8) + b_col + row_in_matrix;
                let dim = comptime!(step as u32 * 16) + b_row;
                let at = (stage_base
                    + key * comptime!(ROW_LINES as u32)
                    + dim / comptime!(LINE as u32)) as usize;
                let b = def.load_matrix::<Vector<f16, Const<8>>, NB>(
                    &key_stage[at..at + 1],
                    MatrixIdent::B,
                    b_regs,
                    key_transposed,
                );
                c = def.execute(&a, &b, &c);
            }
            #[unroll]
            for r in 0..c_regs {
                score[comptime!(tile * c_regs + r)] = c[r];
            }
        }

        // The online softmax, a register row at a time: its max over the unit's cells, then over
        // the quad that shares the row.
        #[unroll]
        for r in 0..c_regs {
            let mut row_max = (-1.0e30f32).runtime();
            #[unroll]
            for tile in 0..score_tiles {
                let cells = score[comptime!(tile * c_regs + r)];
                #[unroll]
                for e in 0..c_size {
                    row_max = max(row_max, cells.extract(e));
                }
            }
            row_max = row_across(&def, row_max, comptime!(r * c_size), Monoid::Max);
            let new_max = max(running_max[r], row_max * scale);
            let correction = (running_max[r] - new_max).exp();
            running_max[r] = new_max;
            let mut row_sum = 0.0f32.runtime();
            #[unroll]
            for tile in 0..score_tiles {
                let cells = score[comptime!(tile * c_regs + r)];
                let p = (cells * Vector::cast_from(scale) - Vector::cast_from(new_max)).exp();
                #[unroll]
                for e in 0..c_size {
                    row_sum += p.extract(e);
                }
                score[comptime!(tile * c_regs + r)] = p;
            }
            sum[r] = sum[r] * correction + row_sum;
            #[unroll]
            for tile in 0..out_tiles {
                acc[comptime!(tile * c_regs + r)] *= Vector::cast_from(correction);
            }
        }

        if comptime!(value_depth == 16) {
            // out += p · v: two score tiles side by side are one `A` fragment of a 16-key step.
            #[unroll]
            for step in 0..comptime!(KEYS / 16) {
                let mut a = Array::<Vector<f16, NA>>::new(a_regs);
                #[unroll]
                for r in 0..a_regs {
                    a[r] = Vector::cast_from(score[comptime!(step * 2 * c_regs + r)]);
                }
                #[unroll]
                for tile in 0..out_tiles {
                    let mut c = Array::<Vector<f32, NC>>::new(c_regs);
                    #[unroll]
                    for r in 0..c_regs {
                        c[r] = acc[comptime!(tile * c_regs + r)];
                    }
                    // `v`'s `(key, dim)` lies where it is staged.
                    let key = comptime!(step as u32 * 16) + b_row + row_in_matrix;
                    let dim = comptime!(tile as u32 * 8) + b_col;
                    let at = (stage_base
                        + key * comptime!(ROW_LINES as u32)
                        + dim / comptime!(LINE as u32)) as usize;
                    let b = def.load_matrix::<Vector<f16, Const<8>>, NB>(
                        &value_stage[at..at + 1],
                        MatrixIdent::B,
                        b_regs,
                        value_transposed,
                    );
                    c = def.execute(&a, &b, &c);
                    #[unroll]
                    for r in 0..c_regs {
                        acc[comptime!(tile * c_regs + r)] = c[r];
                    }
                }
            }
        } else {
            // out += p · v at half depth: one score tile is one `A` fragment of an 8-key step.
            let half = MmaDefinition::<f16, f16, f32>::new(ROWS, 8usize, 8usize);
            let half_b_regs = half.vectors_per_lane(MatrixIdent::B);
            let half_a_regs = half.vectors_per_lane(MatrixIdent::A);
            let (half_row, half_col) = half.position_of_nth(0, 0u32, MatrixIdent::B);
            #[unroll]
            for step in 0..score_tiles {
                let mut a = Array::<Vector<f16, NA>>::new(half_a_regs);
                #[unroll]
                for r in 0..half_a_regs {
                    a[r] = Vector::cast_from(score[comptime!(step * c_regs + r)]);
                }
                #[unroll]
                for tile in 0..out_tiles {
                    let mut c = Array::<Vector<f32, NC>>::new(c_regs);
                    #[unroll]
                    for r in 0..c_regs {
                        c[r] = acc[comptime!(tile * c_regs + r)];
                    }
                    let key = comptime!(step as u32 * 8) + half_row + row_in_matrix;
                    let dim = comptime!(tile as u32 * 8) + half_col;
                    let at = (stage_base
                        + key * comptime!(ROW_LINES as u32)
                        + dim / comptime!(LINE as u32)) as usize;
                    let b = half.load_matrix::<Vector<f16, Const<8>>, NB>(
                        &value_stage[at..at + 1],
                        MatrixIdent::B,
                        half_b_regs,
                        value_transposed,
                    );
                    c = half.execute(&a, &b, &c);
                    #[unroll]
                    for r in 0..c_regs {
                        acc[comptime!(tile * c_regs + r)] = c[r];
                    }
                }
            }
        }
    }

    // The quad's partial sums merged, then each row normalized on its way out.
    let out_base = (head * queries + first_row) * comptime!(DIM as u32);
    #[unroll]
    for r in 0..c_regs {
        let mut total = sum[r];
        total = row_across(&def, total, comptime!(r * c_size), Monoid::Sum);
        let recip = 1.0f32 / total;
        #[unroll]
        for tile in 0..out_tiles {
            let cells = acc[comptime!(tile * c_regs + r)] * Vector::cast_from(recip);
            #[unroll]
            for e in 0..c_size {
                let (row, col) = def.position_of_nth(
                    lane,
                    comptime!(r * c_size + e) as u32,
                    MatrixIdent::Accumulator,
                );
                let at = out_base + row * comptime!(DIM as u32) + comptime!(tile as u32 * 8) + col;
                out[at as usize] = f16::cast_from(cells.extract(e));
            }
        }
    }
}

/// `value` combined under `monoid` over the units holding the same accumulator row as cell `nth`
/// of this unit: a butterfly across the plane over the lane bits that
/// [`MmaDefinition::position_of_nth`] says leave the row in place. Holds for any layout whose row
/// is a function of some of the lane's bits, which [`two_accumulator_tiles_are_one_a_fragment`]
/// checks.
#[cube]
fn row_across(
    def: &MmaDefinition<f16, f16, f32>,
    value: f32,
    #[comptime] nth: usize,
    #[comptime] monoid: Monoid,
) -> f32 {
    // Whether a lane bit moves the row is read at lane 0, where both positions are constants the
    // compiler folds: a bit that moves it costs nothing, its shuffle unused.
    let (row, _) = def.position_of_nth(0u32, nth as u32, MatrixIdent::Accumulator);
    let mut acc = value;
    #[unroll]
    for bit in 0..5u32 {
        let mask = comptime!(1u32 << bit);
        let (other_row, _) = def.position_of_nth(mask, nth as u32, MatrixIdent::Accumulator);
        if other_row == row {
            acc = monoid.combine::<f32>(acc, plane_shuffle_xor(acc, mask));
        }
    }
    acc
}

/// Issue the async copies of one block of keys and values, `[KEYS, DIM]` from `block_base` (in
/// lines), into the stage at `stage_base`, its rows padded to [`ROW_LINES`]: each unit copies
/// whole lines, consecutive units consecutive lines.
#[cube]
fn fill_stage(
    k: &Tensor<Vector<f16, Const<8>>>,
    v: &Tensor<Vector<f16, Const<8>>>,
    key_stage: &mut [Vector<f16, Const<8>>],
    value_stage: &mut [Vector<f16, Const<8>>],
    block_base: u32,
    stage_base: u32,
) {
    let units = comptime!((PLANES * 32) as u32);
    #[unroll]
    for i in 0..comptime!(KEYS * DIM / LINE / (PLANES * 32)) {
        let line = comptime!(i as u32) * units + UNIT_POS;
        let row = line / comptime!((DIM / LINE) as u32);
        let col = line % comptime!((DIM / LINE) as u32);
        let staged = (stage_base + row * comptime!(ROW_LINES as u32) + col) as usize;
        let source = (block_base + line) as usize;
        copy_async(
            &k[source..source + 1],
            &mut key_stage[staged..staged + 1],
            comptime!(LINE as u32),
        );
        copy_async(
            &v[source..source + 1],
            &mut value_stage[staged..staged + 1],
            comptime!(LINE as u32),
        );
    }
}

/// Each unit's cell positions in an `m16n8k16` instruction, written as `(row, col)` pairs: its
/// `A`'s, its `f32` accumulator's, and the `f16` accumulator's of the same shape.
#[cube(launch)]
fn fragment_positions(a: &mut Tensor<u32>, acc: &mut Tensor<u32>, half_acc: &mut Tensor<u32>) {
    let def = MmaDefinition::<f16, f16, f32>::new(ROWS, 8usize, 16usize);
    let half = MmaDefinition::<f16, f16, f16>::new(ROWS, 8usize, 16usize);
    let lane = UNIT_POS_PLANE;
    let a_elems = def.elems_per_lane(MatrixIdent::A);
    let c_elems = def.elems_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for e in 0..a_elems {
        let (row, col) = def.position_of_nth(lane, e as u32, MatrixIdent::A);
        let at = (lane * comptime!(a_elems as u32) + e as u32) * 2;
        a[at as usize] = row;
        a[at as usize + 1] = col;
    }
    #[unroll]
    for e in 0..c_elems {
        let (row, col) = def.position_of_nth(lane, e as u32, MatrixIdent::Accumulator);
        let at = (lane * comptime!(c_elems as u32) + e as u32) * 2;
        acc[at as usize] = row;
        acc[at as usize + 1] = col;
        let (row, col) = half.position_of_nth(lane, e as u32, MatrixIdent::Accumulator);
        half_acc[at as usize] = row;
        half_acc[at as usize + 1] = col;
    }
}

/// The layout facts the register leaf rests on. Cell `e` of an `m16k16` `A` fragment sits where
/// cell `e % half` of the accumulator of `n8` tile `e / half` does, `half` being the accumulator's
/// cells per unit: where it fails, the probabilities are contracted with the wrong keys. An `f16`
/// accumulator keeps its cells where an `f32` one does: where it fails, casting the score in its
/// registers moves its cells. And a row's units differ in lane bits that differ alike on every
/// lane: where it fails, a row reduction combines another row's cells.
#[test]
fn two_accumulator_tiles_are_one_a_fragment() {
    let client = cubecl::test_device().client();
    if !offers_mma(&client) {
        return;
    }
    let (a_elems, c_elems) = (8usize, 4usize);
    let a = client.empty(32 * a_elems * 2 * 4);
    let acc = client.empty(32 * c_elems * 2 * 4);
    let half_acc = client.empty(32 * c_elems * 2 * 4);
    fragment_positions::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(32),
        tensor_arg(a.clone(), &[32 * a_elems * 2]),
        tensor_arg(acc.clone(), &[32 * c_elems * 2]),
        tensor_arg(half_acc.clone(), &[32 * c_elems * 2]),
    );
    let a = u32::from_bytes(&client.read_one_unchecked(a)).to_vec();
    let acc = u32::from_bytes(&client.read_one_unchecked(acc)).to_vec();
    let half_acc = u32::from_bytes(&client.read_one_unchecked(half_acc)).to_vec();
    assert_eq!(
        half_acc, acc,
        "an f16 accumulator's cells sit where an f32 one's do"
    );
    for lane in 0..32 {
        for e in 0..a_elems {
            let (tile, cell) = (e / c_elems, e % c_elems);
            let a_at = (lane * a_elems + e) * 2;
            let c_at = (lane * c_elems + cell) * 2;
            assert_eq!(
                (a[a_at], a[a_at + 1]),
                (acc[c_at], acc[c_at + 1] + 8 * tile as u32),
                "lane {lane}, A cell {e}"
            );
        }
        // Whether a lane bit moves a cell's row is the same on every lane as on lane 0: the row
        // reduction reads it there.
        for e in 0..c_elems {
            let row = |l: usize| acc[(l * c_elems + e) * 2];
            for bit in 0..5 {
                let mask = 1 << bit;
                assert_eq!(
                    row(lane) == row(lane ^ mask),
                    row(0) == row(mask),
                    "lane {lane}, cell {e}, lane bit {bit}"
                );
            }
        }
    }
}

/// The leaf against a host reference over several key blocks, so the running max moves and the
/// output is corrected.
#[test]
fn register_attention_matches_the_reference() {
    let client = cubecl::test_device().client();
    if !offers_mma(&client) {
        return;
    }
    let problem = Problem {
        heads: 8,
        kv_heads: 2,
        queries: 128,
        keys: 4 * KEYS,
    };
    let inputs = problem.inputs(&client);
    let want = problem.reference(&inputs);
    for value_depth in [16, 8] {
        let got = problem.launch(&client, &inputs, value_depth);
        let got = f16::from_bytes(&client.read_one_unchecked(got)).to_vec();
        for (i, (have, want)) in got.iter().zip(&want).enumerate() {
            let err = (have.to_f32() - want).abs();
            assert!(
                err <= 2e-3 + 1e-2 * want.abs(),
                "value depth {value_depth}, cell {i}: got {have}, want {want}"
            );
        }
    }
}

/// The leaf's rate at the prefill shape of a Qwen3-4B layer: 512 queries against 2048 keys,
/// 32 query heads over 8 key-value heads, `d = 128`. Best of a few timed launches after warm ones.
#[test]
#[ignore = "a measurement: run it deliberately, in release"]
fn register_attention_throughput() {
    let client = cubecl::test_device().client();
    if !offers_mma(&client) {
        eprintln!("the device offers no f16 m16n8k16 instruction");
        return;
    }
    let problem = Problem {
        heads: 32,
        kv_heads: 8,
        queries: 512,
        keys: 2048,
    };
    let inputs = problem.inputs(&client);
    let flops = 4.0 * (problem.heads * problem.queries * problem.keys * DIM) as f64;
    for value_depth in [16, 8] {
        for _ in 0..5 {
            problem.launch(&client, &inputs, value_depth);
        }
        cubecl::future::block_on(client.sync()).unwrap();
        let launches = 20;
        let best = (0..5)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..launches {
                    problem.launch(&client, &inputs, value_depth);
                }
                cubecl::future::block_on(client.sync()).unwrap();
                start.elapsed().as_secs_f64() / launches as f64
            })
            .fold(f64::MAX, f64::min);
        eprintln!(
            "REGISTER ATTENTION {}q x {}k, {}:{} heads, d={DIM}, value k={value_depth}: \
             {:8.1} us  {:6.1} TFLOP/s",
            problem.queries,
            problem.keys,
            problem.heads,
            problem.kv_heads,
            best * 1e6,
            flops / best / 1e12
        );
    }
}

/// `repeats` rounds of eight independent `m16n8k16` products per plane on registers alone: the
/// instruction's ceiling on this device, the denominator of the leaf's rate.
#[cube(launch)]
fn mma_ceiling(out: &mut Tensor<f32>, #[comptime] repeats: u32) {
    let def = MmaDefinition::<f16, f16, f32>::new(ROWS, 8usize, 16usize);
    let size!(NA) = def.vector_size(MatrixIdent::A);
    let size!(NB) = def.vector_size(MatrixIdent::B);
    let size!(NC) = def.vector_size(MatrixIdent::Accumulator);
    let a_regs = def.vectors_per_lane(MatrixIdent::A);
    let b_regs = def.vectors_per_lane(MatrixIdent::B);
    let c_regs = def.vectors_per_lane(MatrixIdent::Accumulator);
    let seed = f16::cast_from(UNIT_POS % 3);
    let mut a = Array::<Vector<f16, NA>>::new(a_regs);
    let mut b = Array::<Vector<f16, NB>>::new(b_regs);
    #[unroll]
    for r in 0..a_regs {
        a[r] = Vector::cast_from(seed);
    }
    #[unroll]
    for r in 0..b_regs {
        b[r] = Vector::cast_from(seed);
    }
    let chains = 8usize;
    let mut acc = Array::<Vector<f32, NC>>::new(comptime!(chains * c_regs));
    #[unroll]
    for i in 0..comptime!(chains * c_regs) {
        acc[i] = Vector::cast_from(0.0f32);
    }
    for _ in 0..repeats {
        #[unroll]
        for chain in 0..chains {
            let mut c = Array::<Vector<f32, NC>>::new(c_regs);
            #[unroll]
            for r in 0..c_regs {
                c[r] = acc[comptime!(chain * c_regs + r)];
            }
            c = def.execute(&a, &b, &c);
            #[unroll]
            for r in 0..c_regs {
                acc[comptime!(chain * c_regs + r)] = c[r];
            }
        }
    }
    let mut total = 0.0f32.runtime();
    #[unroll]
    for i in 0..comptime!(chains * c_regs) {
        total += acc[i].extract(0usize);
    }
    out[CUBE_POS * CUBE_DIM as usize + UNIT_POS as usize] = total;
}

/// The `f16` `m16n8k16` instruction's rate over a grid that fills the device, best of a few.
#[test]
#[ignore = "a measurement: run it deliberately, in release"]
fn mma_ceiling_throughput() {
    let client = cubecl::test_device().client();
    if !offers_mma(&client) {
        return;
    }
    let (cubes, planes, repeats) = (1024u32, 4u32, 4096u32);
    let out = client.empty((cubes * planes * 32) as usize * 4);
    let launch = || {
        mma_ceiling::launch(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_2d(32, planes),
            tensor_arg(out.clone(), &[(cubes * planes * 32) as usize]),
            repeats,
        )
    };
    let flops = 2.0 * (16 * 8 * 16) as f64 * 8.0 * repeats as f64 * (cubes * planes) as f64;
    for _ in 0..3 {
        launch();
    }
    cubecl::future::block_on(client.sync()).unwrap();
    let best = (0..5)
        .map(|_| {
            let start = Instant::now();
            launch();
            cubecl::future::block_on(client.sync()).unwrap();
            start.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min);
    eprintln!(
        "MMA CEILING f16 m16n8k16: {:6.1} TFLOP/s",
        flops / best / 1e12
    );
}

/// One attention: its heads, and the queries and keys of each.
#[derive(Clone, Copy)]
struct Problem {
    heads: usize,
    kv_heads: usize,
    queries: usize,
    keys: usize,
}

/// The inputs of one attention, on the host and on the device.
struct Inputs {
    q: Vec<f16>,
    k: Vec<f16>,
    v: Vec<f16>,
    device: [cubecl::server::Handle; 3],
}

impl Problem {
    fn scale(&self) -> f32 {
        1.0 / (DIM as f32).sqrt()
    }

    /// Values spread over a few units, so the scores' rows hold maxima well apart and the
    /// running max moves between key blocks.
    fn inputs(&self, client: &Client) -> Inputs {
        let values = |len: usize, seed: usize| -> Vec<f16> {
            (0..len)
                .map(|i| {
                    let x = (i.wrapping_mul(2654435761).wrapping_add(seed * 97)) % 2001;
                    f16::from_f32((x as f32 / 1000.0 - 1.0) * 2.0)
                })
                .collect()
        };
        let q = values(self.heads * self.queries * DIM, 1);
        let k = values(self.kv_heads * self.keys * DIM, 2);
        let v = values(self.kv_heads * self.keys * DIM, 3);
        let upload = |values: &[f16]| client.create_from_slice(f16::as_bytes(values));
        Inputs {
            device: [upload(&q), upload(&k), upload(&v)],
            q,
            k,
            v,
        }
    }

    /// Launch the leaf, the value contracting the keys `value_depth` (16 or 8) a step.
    fn launch(
        &self,
        client: &Client,
        inputs: &Inputs,
        value_depth: usize,
    ) -> cubecl::server::Handle {
        let [q, k, v] = inputs.device.clone();
        let out = client.empty(self.heads * self.queries * DIM * 2);
        let rows = self.heads * self.queries * DIM;
        let kv = self.kv_heads * self.keys * DIM;
        register_attention::launch(
            client,
            CubeCount::Static(
                (self.queries / (ROWS * PLANES)) as u32,
                self.heads as u32,
                1,
            ),
            CubeDim::new_2d(32, PLANES as u32),
            tensor_arg(q, &[rows]),
            tensor_arg(k, &[kv / LINE]),
            tensor_arg(v, &[kv / LINE]),
            tensor_arg(out.clone(), &[rows]),
            self.queries as u32,
            self.keys as u32,
            self.scale(),
            (self.heads / self.kv_heads) as u32,
            value_depth,
        );
        out
    }

    /// The attention in `f64` over the `f16` inputs.
    fn reference(&self, inputs: &Inputs) -> Vec<f32> {
        let group = self.heads / self.kv_heads;
        let mut out = vec![0f32; self.heads * self.queries * DIM];
        let mut scores = vec![0f64; self.keys];
        for h in 0..self.heads {
            let kv = h / group;
            for i in 0..self.queries {
                let q = &inputs.q[(h * self.queries + i) * DIM..][..DIM];
                for (j, s) in scores.iter_mut().enumerate() {
                    let k = &inputs.k[(kv * self.keys + j) * DIM..][..DIM];
                    *s = q
                        .iter()
                        .zip(k)
                        .map(|(a, b)| a.to_f64() * b.to_f64())
                        .sum::<f64>()
                        * self.scale() as f64;
                }
                let max = scores.iter().cloned().fold(f64::MIN, f64::max);
                let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for d in 0..DIM {
                    let o: f64 = scores
                        .iter()
                        .enumerate()
                        .map(|(j, s)| {
                            (s - max).exp() * inputs.v[(kv * self.keys + j) * DIM + d].to_f64()
                        })
                        .sum();
                    out[(h * self.queries + i) * DIM + d] = (o / total) as f32;
                }
            }
        }
        out
    }
}

fn tensor_arg(handle: cubecl::server::Handle, shape: &[usize]) -> TensorArg {
    unsafe { TensorArg::from_raw_parts(handle, [1].into(), shape.to_vec().into()) }
}

fn offers_mma(client: &Client) -> bool {
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    client.features().matmul.mma.contains(&MmaConfig {
        a_type: ElemType::Float(FloatKind::F16),
        b_type: ElemType::Float(FloatKind::F16),
        cd_type: ElemType::Float(FloatKind::F32),
        m: 16,
        n: 8,
        k: 16,
    })
}
