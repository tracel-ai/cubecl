//! A manual-mma fragment deeper along `k` than its instruction: the registers of several
//! instructions, which a contraction issues one after another into the same accumulator.
//!
//! A sum does not care which value of `k` meets which slot, only that both factors hold the same
//! one there. So such a fragment lays `k` out in **lane runs**: each unit's slots, across all of
//! the fragment's instructions, hold one contiguous run of `k`. A unit then reads its share in
//! whole served loads rather than a pair of cells an instruction, and a packed operand decodes
//! and scales a whole word in registers. A contraction loads both its factors in lane runs or
//! both a step at a time ([`DeepLoad`]); a fragment one instruction deep keeps the instruction's
//! own order.
//!
//! The map is read off the 16-bit `m16n8k16` layout: unit `u` holds the cells of quad `u % 4`
//! along `k`, a pair of adjacent cells a register, its first pair from the instruction's first
//! eight values of `k` and its second from the last eight. [`lays_out_lane_runs`] admits the
//! targets whose layout this is, and the component test checks the map on every unit.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2,
    ir::types::Fp8Format,
    prelude::*,
    quant::scheme::QuantValue,
};

use super::block_scaled::{E2M1_PER_WORD, E2M1_SCALE_BLOCK};
use super::widths::*;
use crate::*;

/// Slots of one instruction's `k` a unit holds, on the 16-bit `m16n8k16` layout: four of sixteen.
const SLOTS_PER_UNIT: usize = 4;

/// Adjacent slots one register holds: a pair of 16-bit values.
const SLOTS_PER_REGISTER: usize = 2;

/// Units sharing one row of `A` or column of `B`, each holding its own quad of `k`.
const UNITS_PER_ROW: u32 = 4;

/// Rows between the two an `A` unit holds, `g` and `g + 8`.
const HALF_ROWS: u32 = 8;

/// The depth along `k` one instruction of an `m × n` fragment of `T` contracts: `k` itself where
/// the device offers that instruction, else the deepest one it offers that divides `k`, else `k`,
/// which the instruction's expansion refuses.
// `T` is read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
pub(crate) fn instruction_depth<T: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(usize) {
    intrinsic!(|scope| {
        let element = T::elem_type(scope);
        let Some(properties) = scope.state().device_properties.as_ref() else {
            return k;
        };
        let depths = properties
            .features
            .matmul
            .mma
            .iter()
            .filter(|config| {
                config.a_type == element
                    && config.b_type == element
                    && (config.m as usize, config.n as usize) == (m, n)
            })
            .map(|config| config.k as usize)
            .collect::<Vec<_>>();
        if depths.contains(&k) {
            return k;
        }
        depths
            .into_iter()
            .filter(|&depth| k.is_multiple_of(depth))
            .max()
            .unwrap_or(k)
    })
}

/// Instructions one `m × n × k` fragment of `T` holds along `k`: one, unless it is deeper than its
/// instruction, when it holds `k` in lane runs.
#[cube]
pub(crate) fn lane_run_steps<T: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(usize) {
    let depth = instruction_depth::<T>(m, n, k);
    comptime!(k / depth)
}

/// Whether the target lays a 16-bit `m16n8k16` instruction's registers out as the lane-run map
/// reads them: a 32-unit plane, `A` along its rows and `B` along its columns, every cell held
/// once.
#[cube]
pub(crate) fn lays_out_lane_runs() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let mma = &scope.state().target_properties.mma;
        mma.const_plane_size == 32
            && mma.register_layout_a == MatrixLayout::RowMajor
            && mma.register_layout_b == MatrixLayout::ColMajor
            && mma.register_duplication_a == 1
            && mma.register_duplication_b == 1
    })
}

/// Why `src`, a window of an `m × n × k` fragment of `T` in role `ident`, cannot load in lane
/// runs; `None` where it can.
///
/// It can where the fragment is deeper than its instruction, a 16-bit `m16n8k16` one, on a target
/// that [`lays_out_lane_runs`]; its window runs along `k`; and it is plain, or `e2m1` codes under
/// one level of scales whose product with a code is exact in 16 bits, NVFP4's `e4m3` or a power
/// of two. A coarser level is the product's to apply to the accumulator.
#[cube]
pub(crate) fn lane_runs_refusal<T: Numeric>(
    src: &Tile,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
) -> comptime_type!(Option<&'static str>) {
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let laid_out = lays_out_lane_runs();
    let bytes = T::size().comptime();
    let packing = src.packing();
    let levels = src.factor_levels();
    let stored = src.innermost_factor().inner_stored();
    let along_k = comptime!(match ident {
        MatrixIdent::A => layout == MatrixLayout::RowMajor,
        MatrixIdent::B => layout == MatrixLayout::ColMajor,
        MatrixIdent::Accumulator => false,
    });
    let e2m1 = comptime!(matches!(
        packing,
        Packing::Packed {
            field: Field::Quant(QuantValue::E2M1) | Field::ConvertedE2M1
        }
    ));
    let exact_scales = comptime!(matches!(
        stored,
        Some(Packing::Packed {
            field: Field::Fp8(Fp8Format::E4M3 | Fp8Format::UE8M0)
        })
    ));
    comptime!(if k <= depth {
        Some("the fragment is one instruction deep, which keeps the instruction's own order")
    } else if (m, n, depth) != (16, 8, 16) || bytes != 2 || !laid_out {
        Some("lane runs read the 16-bit m16n8k16 layout, which this target or element is not")
    } else if !along_k {
        Some("a window loads in lane runs along `k`: an A row-major, a B col-major")
    } else if packing == Packing::Plain && levels == 0 {
        None
    } else if !e2m1 {
        Some("a packed or scaled window loads in lane runs only as e2m1 codes the device converts")
    } else if levels != 1 || !exact_scales {
        Some(
            "e2m1 codes load in lane runs under one level of e4m3 or power-of-two scales, whose \
             product with a code is exact in 16 bits",
        )
    } else {
        None
    })
}

/// A unit's run of a lane-run window: how long it is, which window it reads, and the register
/// each of its slots lands in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LaneRun {
    /// The fragment's role.
    ident: MatrixIdent,
    /// Values of `k` the run holds: [`SLOTS_PER_UNIT`] of each instruction.
    len: usize,
    /// Registers one instruction holds in this role.
    per_step: usize,
    /// The window's rows, its free edge: fewer than the fragment's where a product of fewer rows
    /// leaves it short, the rest read as zeros by the checked view.
    rows: usize,
    /// The window's `k`, however many axes spell it.
    k: usize,
}

impl LaneRun {
    /// Rows of the window the plane's units cover: an `A`'s sixteen, `g` and `g + 8` for each of
    /// the eight `g`, a `B`'s eight columns.
    fn covered_rows(&self) -> usize {
        self.halves() * HALF_ROWS as usize
    }

    /// Rows of the window one unit holds: an `A`'s `g` and `g + 8`, a `B`'s one column.
    fn halves(&self) -> usize {
        match self.ident {
            MatrixIdent::A => 2,
            MatrixIdent::B => 1,
            MatrixIdent::Accumulator => unreachable!("an accumulator holds no slots of `k`"),
        }
    }

    /// The register holding cell `cell` of the run, in half `half` of the unit's rows. Cell
    /// `4s + j` is slot `j` of instruction `s`; an `A` holds its rows in alternate registers, its
    /// first pair of `k` in registers 0 and 1, a `B` a pair a register.
    fn register(&self, cell: usize, half: usize) -> usize {
        let (step, pair) = (
            cell / SLOTS_PER_UNIT,
            cell % SLOTS_PER_UNIT / SLOTS_PER_REGISTER,
        );
        let within = match self.ident {
            MatrixIdent::A => pair * 2 + half,
            MatrixIdent::B => pair,
            MatrixIdent::Accumulator => unreachable!("an accumulator holds no slots of `k`"),
        };
        step * self.per_step + within
    }
}

/// Load `fragment`, an `m × n × k` fragment of role `ident` deeper than its instruction, from
/// `src` in lane runs: each unit reads the run of its quad along each of its rows of the window.
#[cube]
pub(crate) fn load_lane_runs<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
) {
    let refusal = lane_runs_refusal::<T>(src, ident, layout, shape);
    comptime!(if let Some(why) = refusal {
        panic!("MmaData::load: a fragment deeper than its instruction loads in lane runs: {why}")
    });
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let per_step = MmaDefinition::<T, T, T>::new(m, n, depth).vectors_per_lane(ident);
    let space = comptime!(&src.place.space);
    let run = comptime!(LaneRun {
        ident,
        len: SLOTS_PER_UNIT * (k / depth),
        per_step,
        rows: space
            .axes()
            .map(|axis| space.extent(axis))
            .product::<usize>()
            / k,
        k,
    });
    let row = UNIT_POS_PLANE / UNITS_PER_ROW;
    let first = (UNIT_POS_PLANE % UNITS_PER_ROW) * comptime!(run.len as u32);
    let packing = src.packing();
    match comptime!(packing) {
        Packing::Plain => load_plain_run::<T, N>(src, fragment, row, first, run),
        Packing::Packed { .. } => load_e2m1_run::<T, N>(src, fragment, row, first, run),
    }
}

/// [`load_lane_runs`] of a plain window: the run read a served line at a time.
#[cube]
fn load_plain_run<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    row: u32,
    first: u32,
    #[comptime] run: LaneRun,
) {
    let width = src.vector_size();
    comptime!(assert!(
        run.len.is_multiple_of(width),
        "MmaData::load: a {width}-wide line runs past a unit's {}-cell run of `k`; serve the \
         operand at most {} wide",
        run.len,
        run.len,
    ));
    let size!(W) = width;
    let view = src.fragment_matrix_packed::<T, W>(comptime!(run.rows), comptime!(run.k));
    let lines = comptime!(run.len / width);
    // A window of fewer rows than the units cover, a decode's rows staged as they are, holds no
    // cells past them: those units read zeros rather than past its buffer.
    let short = comptime!(run.rows < run.covered_rows());
    #[unroll]
    for half in 0..comptime!(run.halves()) {
        let at = row + comptime!(half as u32) * HALF_ROWS;
        let mut cells = Array::<Vector<T, W>>::new(lines);
        #[unroll]
        for l in 0..lines {
            cells[l] = Vector::<T, W>::new(T::from_int(0));
            if !short || at < comptime!(run.rows as u32) {
                cells[l] = view.read((at, first / comptime!(width as u32) + comptime!(l as u32)));
            }
        }
        #[unroll]
        for p in 0..comptime!(run.len / SLOTS_PER_REGISTER) {
            let cell = comptime!(p * SLOTS_PER_REGISTER);
            let mut pair = Vector::<T, N>::empty();
            #[unroll]
            for e in 0..SLOTS_PER_REGISTER {
                let c = comptime!(cell + e);
                pair.insert(e, cells[comptime!(c / width)].extract(comptime!(c % width)));
            }
            fragment[comptime!(run.register(cell, half))] = pair;
        }
    }
}

/// [`load_lane_runs`] of `e2m1` codes under one level of block scales: the run read as the stored
/// words it is, each word decoded to eight values and multiplied by its block's scale in `T`,
/// exact under [`lane_runs_refusal`]'s conditions.
#[cube]
fn load_e2m1_run<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    row: u32,
    first: u32,
    #[comptime] run: LaneRun,
) {
    let space = comptime!(src.place.space.clone());
    let axes = comptime!(
        MatrixAxes::new(&space, run.rows, run.k).unwrap_or_else(|no| {
            panic!(
                "MmaData::load: a lane-run window is a {} x {} matrix: {no:?}",
                run.rows, run.k
            )
        })
    );
    let load = src.vector_tile();
    let values = comptime!(load.values());
    // A load lies along `k` alone: one that also spans rows would place another row's codes in
    // this unit's run.
    let along_k = comptime!(
        load.space()
            .axes()
            .all(|axis| (axes.col_split..space.rank()).any(|p| space.axis_at(p) == axis))
    );
    comptime!(assert!(
        along_k,
        "MmaData::load: a unit's run of `k` reads loads along `k` alone, and this window's load \
         spans {:?}",
        load.space()
    ));
    comptime!(assert!(
        values.is_multiple_of(E2M1_PER_WORD)
            && run.len.is_multiple_of(values)
            && run.len.is_multiple_of(E2M1_SCALE_BLOCK),
        "MmaData::load: a unit's {}-value run of `k` reads whole loads and whole scale blocks, \
         and this window is read {values} values a load",
        run.len,
    ));
    let size!(WP) = comptime!(values / E2M1_PER_WORD);
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let factor = src.innermost_factor();
    #[unroll]
    for half in 0..comptime!(run.halves()) {
        let at = row + comptime!(half as u32) * HALF_ROWS;
        let mut scales = Array::<T>::new(comptime!(run.len / E2M1_SCALE_BLOCK));
        #[unroll]
        for b in 0..comptime!(run.len / E2M1_SCALE_BLOCK) {
            let col = first + comptime!((b * E2M1_SCALE_BLOCK) as u32);
            let coords = TileMatrix::value_coords(at, col, 0usize, &space, axes, 1usize);
            scales[b] = T::cast_from(factor.at_coords(&coords, comptime!(space.clone())));
        }
        #[unroll]
        for l in 0..comptime!(run.len / values) {
            let col = first + comptime!((l * values) as u32);
            let coords = TileMatrix::value_coords(at, col, 0usize, &space, axes, 1usize);
            let line = words.read(load.index(&coords, &space));
            #[unroll]
            for w in 0..comptime!(values / E2M1_PER_WORD) {
                let word = comptime!(l * values + w * E2M1_PER_WORD);
                let codes = Vector::<e2m1x2, Const<4>>::reinterpret(line.extract(w));
                let decoded = Vector::<T, Const<8>>::cast_from(codes)
                    * Vector::<T, Const<8>>::new(scales[comptime!(word / E2M1_SCALE_BLOCK)]);
                #[unroll]
                for p in 0..comptime!(E2M1_PER_WORD / SLOTS_PER_REGISTER) {
                    let mut pair = Vector::<T, N>::empty();
                    #[unroll]
                    for e in 0..SLOTS_PER_REGISTER {
                        pair.insert(e, decoded.extract(comptime!(p * SLOTS_PER_REGISTER + e)));
                    }
                    let cell = comptime!(word + p * SLOTS_PER_REGISTER);
                    fragment[comptime!(run.register(cell, half))] = pair;
                }
            }
        }
    }
}

/// `acc += lhs · rhs` over fragments `steps` instructions of `def` deep, issued in turn into the
/// one accumulator.
#[cube]
pub(crate) fn mma_execute_lane_runs<L: Numeric, R: Numeric, A: Numeric>(
    def: &MmaDefinition<L, R, A>,
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] steps: usize,
) {
    let a_registers = def.vectors_per_lane(MatrixIdent::A);
    let b_registers = def.vectors_per_lane(MatrixIdent::B);
    #[unroll]
    for s in 0..steps {
        let mut a = Array::<Vector<L, NL>>::new(a_registers);
        let mut b = Array::<Vector<R, NR>>::new(b_registers);
        #[unroll]
        for i in 0..a_registers {
            a[i] = lhs[comptime!(s * a_registers + i)];
        }
        #[unroll]
        for i in 0..b_registers {
            b[i] = rhs[comptime!(s * b_registers + i)];
        }
        def.execute_inplace(&a, &b, acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run two instructions deep on the 16-bit `m16n8k16` layout.
    fn two_steps(ident: MatrixIdent, per_step: usize) -> LaneRun {
        LaneRun {
            ident,
            len: 2 * SLOTS_PER_UNIT,
            per_step,
            rows: 16,
            k: 32,
        }
    }

    /// Every register of every instruction receives exactly one pair of the run, an `A`'s first
    /// pair of `k` in registers 0 (rows `g`) and 1 (rows `g + 8`).
    #[test]
    fn each_register_takes_one_pair_of_the_run() {
        for (ident, per_step) in [(MatrixIdent::A, 4), (MatrixIdent::B, 2)] {
            let run = two_steps(ident, per_step);
            let mut registers: Vec<_> = (0..run.halves())
                .flat_map(|half| {
                    (0..run.len)
                        .step_by(SLOTS_PER_REGISTER)
                        .map(move |cell| run.register(cell, half))
                })
                .collect();
            registers.sort();
            assert_eq!(
                registers,
                (0..2 * per_step).collect::<Vec<_>>(),
                "{ident:?}"
            );
        }
        let a = two_steps(MatrixIdent::A, 4);
        assert_eq!(
            (a.register(0, 0), a.register(0, 1), a.register(2, 0)),
            (0, 1, 2)
        );
    }
}
