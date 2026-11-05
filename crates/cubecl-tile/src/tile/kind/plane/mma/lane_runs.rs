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
//! The map is read off the target's 16-bit instruction ([`LaneMap`]), adjacent cells of `k` a
//! register vector: a pair on NVIDIA, eight on RDNA3. On NVIDIA's `m16n8k16` unit `u` holds the
//! cells of quad `u % 4` along `k`, its first pair from the instruction's first eight values of
//! `k` and its second from the last eight. On RDNA3's `16 × 16`, sixteen or thirty-two deep, unit
//! `u` holds every value of `k` of its row, the plane's two halves holding the same rows.
//! [`LaneMap::new`] admits the targets whose layout is one of these, and the component test
//! checks the map on every unit.
//!
//! A window of fewer rows than the units cover leaves the accumulator's cells past its rows
//! holding whatever those units read: the rows a store drops, and that a fused consumer of the
//! accumulator masks as it masks any cell past the product's edge.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    ir::types::Fp8Format,
    prelude::*,
};

use super::widths::*;
use crate::*;

/// `e2m1` values one stored word holds.
pub(crate) const E2M1_PER_WORD: usize = 8;

/// Values of `k` a run reads one scale for: sixteen, the narrowest block a format scales, so
/// a run under any block of a multiple of it reads each value's scale.
pub(crate) const SCALE_READ: usize = 16;

/// How a target's 16-bit manual-mma instruction lays `k` out across a plane's units: unit `u`
/// holds part `u % units_per_row` of the instruction's `k` along row
/// `(u / units_per_row) % rows_spanned` of an `A`, and `rows_spanned` further on where it holds a
/// second, or along that column of a `B`. Units past `units_per_row × rows_spanned` hold what the
/// first ones do.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct LaneMap {
    /// The instruction's `m × n`.
    edges: (usize, usize),
    /// The depths along `k` it lays out so: a deeper one holds its `k` as the shallowest does.
    depths: &'static [usize],
    /// Units sharing one row of `A` or column of `B`, each holding its own part of `k`.
    units_per_row: usize,
    /// Rows the units span: an `A` unit's second row lies this far past its first.
    rows_spanned: usize,
    /// Rows of the window one `A` unit holds; a `B` unit holds one column.
    rows_per_a_unit: usize,
}

impl LaneMap {
    /// Units of the plane a lane map lays out.
    const PLANE_UNITS: usize = 32;

    /// NVIDIA's `m16n8k16`: unit `u` holds quad `u % 4` of `k`, an `A`'s rows `u / 4` and
    /// `u / 4 + 8`, a `B`'s column `u / 4`.
    pub(crate) const M16N8K16: LaneMap = LaneMap {
        edges: (16, 8),
        depths: &[16],
        units_per_row: 4,
        rows_spanned: 8,
        rows_per_a_unit: 2,
    };

    /// RDNA3's `16 × 16`, sixteen or thirty-two deep: unit `u` holds every value of `k` along row
    /// `u % 16` of an `A`, or that column of a `B`, in order, the plane's two halves holding the
    /// same.
    pub(crate) const RDNA3_16X16: LaneMap = LaneMap {
        edges: (16, 16),
        depths: &[16, 32],
        units_per_row: 1,
        rows_spanned: 16,
        rows_per_a_unit: 1,
    };

    /// Slots of one `depth`-deep instruction's `k` a unit holds.
    fn slots_per_unit(&self, depth: usize) -> usize {
        depth / self.units_per_row
    }

    /// Whether the plane's later units hold what its first ones do.
    fn duplicated(&self) -> bool {
        self.units_per_row * self.rows_spanned < Self::PLANE_UNITS
    }
}

#[cube]
impl LaneMap {
    /// The target's lane map of its `m × n` 16-bit instruction `depth` deep, where it lays that
    /// one out: a 32-unit plane of 32-bit registers, `A` along its rows and `B` along its columns,
    /// every cell held once (NVIDIA's `m16n8k16`) or by both halves of the plane (RDNA3's
    /// `16 × 16`). The duplication says which layout the target's is, the shape whether this
    /// instruction is one it lays out.
    pub(crate) fn new(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] depth: usize,
    ) -> comptime_type!(Option<LaneMap>) {
        intrinsic!(|scope| {
            let mma = &scope.state().target_properties.mma;
            let laid_out = mma.const_plane_size as usize == LaneMap::PLANE_UNITS
                && mma.register_size_bits == 32
                && mma.register_layout_a == MatrixLayout::RowMajor
                && mma.register_layout_b == MatrixLayout::ColMajor;
            let map = match (mma.register_duplication_a, mma.register_duplication_b) {
                (1, 1) => LaneMap::M16N8K16,
                (2, 2) => LaneMap::RDNA3_16X16,
                _ => return None,
            };
            (laid_out && map.edges == (m, n) && map.depths.contains(&depth)).then_some(map)
        })
    }
}

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

/// Why `src`, a window of an `m × n × k` fragment of `T` in role `ident`, cannot load in lane
/// runs; `None` where it can.
///
/// It can where the fragment is deeper than its instruction, a 16-bit one whose shape is the
/// target's [`LaneMap`]; its window runs along `k`; and it is plain, or `e2m1` codes under
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
    let map = LaneMap::new(m, n, depth);
    let bytes = T::size().comptime();
    let packing = src.packing();
    let levels = src.factor_levels();
    let stored = src.innermost_factor().inner_stored();
    let along_k = comptime!(match ident {
        MatrixIdent::A => layout == MatrixLayout::RowMajor,
        MatrixIdent::B => layout == MatrixLayout::ColMajor,
        MatrixIdent::Accumulator => false,
    });
    let e2m1 = comptime!(matches!(packing, Packing::Packed { field } if field.holds_e2m1()));
    let exact_scales = comptime!(matches!(
        stored,
        Some(Packing::Packed {
            field: Field::Fp8(Fp8Format::E4M3 | Fp8Format::UE8M0)
        })
    ));
    comptime!(if k <= depth {
        Some("the fragment is one instruction deep, which keeps the instruction's own order")
    } else if bytes != 2 || map.is_none() {
        Some(
            "lane runs read a 16-bit m16n8k16 or RDNA3 16x16 layout, which this target or \
             element is not",
        )
    } else if !along_k {
        Some("a window loads in lane runs along `k`: an A row-major, a B col-major")
    } else if packing.as_stored() && levels == 0 {
        None
    } else if !e2m1 {
        Some("a packed or scaled window loads in lane runs only as e2m1 codes")
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
    /// The target's layout of one instruction.
    map: LaneMap,
    /// The instruction's depth along `k`.
    depth: usize,
    /// Values of `k` the run holds: [`LaneMap::slots_per_unit`] of each instruction.
    len: usize,
    /// Registers one instruction holds in this role.
    registers_per_step: usize,
    /// Adjacent cells of `k` one register holds: a pair on `m16n8k16`, eight on RDNA3.
    cells_per_register: usize,
    /// The window's rows, its free edge: fewer than the units cover where a product of fewer rows
    /// leaves it short, the units past them reading its last row again
    /// ([`LaneRun::window_row`]). What they read lands in accumulator cells past the product's
    /// edge, which hold no value of it.
    rows: usize,
    /// The window's `k`, however many axes spell it.
    k: usize,
}

impl LaneRun {
    /// Rows of the window the plane's units cover: on `m16n8k16` an `A`'s sixteen, `g` and
    /// `g + 8` for each of the eight `g`, a `B`'s eight columns; on RDNA3 sixteen of either.
    fn covered_rows(&self) -> usize {
        self.rows_per_unit() * self.map.rows_spanned
    }

    /// Whether the window holds fewer rows than the units cover.
    fn short(&self) -> bool {
        self.rows < self.covered_rows()
    }

    /// Rows of the window one unit holds: an `m16n8k16` `A`'s `g` and `g + 8`, one otherwise.
    fn rows_per_unit(&self) -> usize {
        match self.ident {
            MatrixIdent::A => self.map.rows_per_a_unit,
            MatrixIdent::B => 1,
            MatrixIdent::Accumulator => unreachable!("an accumulator holds no slots of `k`"),
        }
    }

    /// The register holding cell `cell` of the run along the unit's held row `held_row`, the first
    /// of the [`cells_per_register`](Self::cells_per_register) adjacent cells it holds. Cell
    /// `S s + j` is slot `j` of instruction `s`, `S` the slots a unit holds of one; a unit holding
    /// two rows holds them in alternate registers, its first pair of `k` in registers 0 and 1.
    fn register(&self, cell: usize, held_row: usize) -> usize {
        let slots = self.map.slots_per_unit(self.depth);
        let (step, vector) = (cell / slots, cell % slots / self.cells_per_register);
        step * self.registers_per_step + vector * self.rows_per_unit() + held_row
    }
}

#[cube]
impl LaneRun {
    /// The run of `src`, a window of an `m × n × k` fragment of `T` in role `ident` that
    /// [`lane_runs_refusal()`] admits.
    fn new<T: Numeric>(
        src: &Tile,
        #[comptime] ident: MatrixIdent,
        #[comptime] shape: (usize, usize, usize),
    ) -> comptime_type!(LaneRun) {
        let (m, n, k) = comptime!(shape);
        let depth = instruction_depth::<T>(m, n, k);
        let def = MmaDefinition::<T, T, T>::new(m, n, depth);
        let registers_per_step = def.vectors_per_lane(ident);
        let cells_per_register = def.vector_size(ident);
        let map = LaneMap::new(m, n, depth);
        let space = comptime!(&src.place.space);
        comptime!({
            let map = map.expect("lane_runs_refusal admits a target with a lane map alone");
            let cells = space
                .axes()
                .map(|axis| space.extent(axis))
                .product::<usize>();
            assert!(
                cells >= k,
                "MmaData::load: a lane-run window holds {cells} cells, fewer than one row of its \
                 {k} values of `k`"
            );
            let slots = map.slots_per_unit(depth);
            assert!(
                slots.is_multiple_of(cells_per_register),
                "MmaData::load: a {cells_per_register}-cell register runs past a unit's {slots} \
                 slots of one instruction"
            );
            LaneRun {
                ident,
                map,
                depth,
                len: slots * (k / depth),
                registers_per_step,
                cells_per_register,
                rows: cells / k,
                k,
            }
        })
    }

    /// The first value of `k` of this unit's run: its part of the instruction's `k`.
    fn first(#[comptime] run: LaneRun) -> u32 {
        let units_per_row = comptime!(run.map.units_per_row as u32);
        (UNIT_POS_PLANE % units_per_row) * comptime!(run.len as u32)
    }

    /// The window row this unit reads for its held row `held_row`. Past a short window's rows the
    /// unit reads its last row again rather than branch around the read, so the read stays in
    /// the block the contraction issues from. A row of the instruction meets only its own row of
    /// the accumulator, which lies past the product's rows too ([`LaneRun::rows`]).
    fn window_row(#[comptime] run: LaneRun, #[comptime] held_row: usize) -> u32 {
        let row = UNIT_POS_PLANE / comptime!(run.map.units_per_row as u32);
        let row = if comptime!(run.map.duplicated()) {
            row % comptime!(run.map.rows_spanned as u32)
        } else {
            row
        };
        let at = row + comptime!((held_row * run.map.rows_spanned) as u32);
        if comptime!(run.short()) {
            at.min(comptime!(run.rows as u32 - 1))
        } else {
            at
        }
    }
}

/// Load `fragment`, an `m × n × k` fragment of role `ident` deeper than its instruction, from
/// `src` in lane runs: each unit reads its run along each of its rows of the window.
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
    let run = LaneRun::new::<T>(src, ident, shape);
    let packing = src.packing();
    match comptime!(packing) {
        Packing::Plain | Packing::Words { .. } => load_plain_run::<T, N>(src, fragment, run),
        Packing::Packed { .. } => load_e2m1_run::<T, N>(src, fragment, run),
    }
}

/// [`load_lane_runs()`] of a plain window: the run read a served line at a time.
#[cube]
fn load_plain_run<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
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
    let first = LaneRun::first(run);
    #[unroll]
    for held_row in 0..comptime!(run.rows_per_unit()) {
        let at = LaneRun::window_row(run, held_row);
        let mut cells = Array::<Vector<T, W>>::new(lines);
        #[unroll]
        for l in 0..lines {
            cells[l] = view.read((at, first / comptime!(width as u32) + comptime!(l as u32)));
        }
        #[unroll]
        for p in 0..comptime!(run.len / run.cells_per_register) {
            let cell = comptime!(p * run.cells_per_register);
            let mut vector = Vector::<T, N>::empty();
            #[unroll]
            for e in 0..comptime!(run.cells_per_register) {
                let c = comptime!(cell + e);
                vector.insert(e, cells[comptime!(c / width)].extract(comptime!(c % width)));
            }
            fragment[comptime!(run.register(cell, held_row))] = vector;
        }
    }
}

/// [`load_lane_runs()`] of `e2m1` codes under one level of block scales: the run read as the stored
/// words it is, each word decoded to eight values and multiplied by its block's scale in `T`,
/// exact under [`lane_runs_refusal()`]'s conditions.
#[cube]
fn load_e2m1_run<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
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
            && run.len.is_multiple_of(SCALE_READ),
        "MmaData::load: a unit's {}-value run of `k` reads whole loads and whole scale blocks, \
         and this window is read {values} values a load",
        run.len,
    ));
    let size!(WP) = comptime!(values / E2M1_PER_WORD);
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let factor = src.innermost_factor();
    // The codes as the device reads them ([`Field::read_on`]): cast as `e2m1x2` only where the
    // device converts them, decoded from the word's bits everywhere else. Each value is whole,
    // since its block's scale is all this run multiplies it by.
    let packing = src.packing();
    let field = comptime!(match packing {
        Packing::Packed {
            field: Field::Placed { value, .. },
        } => Field::Placed {
            value,
            lifted: true,
        },
        Packing::Packed { field } => field,
        Packing::Plain | Packing::Words { .. } =>
            panic!("MmaData::load: an `e2m1` run reads packed words"),
    });
    let first = LaneRun::first(run);
    #[unroll]
    for held_row in 0..comptime!(run.rows_per_unit()) {
        let at = LaneRun::window_row(run, held_row);
        let mut scales = Array::<T>::new(comptime!(run.len / SCALE_READ));
        #[unroll]
        for b in 0..comptime!(run.len / SCALE_READ) {
            let col = first + comptime!((b * SCALE_READ) as u32);
            let coords = TileMatrix::value_coords(at, col, 0usize, &space, axes, 1usize);
            scales[b] = T::cast_from(factor.at_coords(
                &coords,
                comptime!(space.clone()),
                comptime!(Guard::Checked),
            ));
        }
        #[unroll]
        for l in 0..comptime!(run.len / values) {
            let col = first + comptime!((l * values) as u32);
            let coords = TileMatrix::value_coords(at, col, 0usize, &space, axes, 1usize);
            let line = words.read(load.index(&coords, &space));
            #[unroll]
            for w in 0..comptime!(values / E2M1_PER_WORD) {
                let word = comptime!(l * values + w * E2M1_PER_WORD);
                let codes = unpack_line::<T, Const<1>, Const<8>>(
                    Vector::new(line.extract(w)),
                    comptime!(field),
                );
                let decoded =
                    codes * Vector::<T, Const<8>>::new(scales[comptime!(word / SCALE_READ)]);
                #[unroll]
                for p in 0..comptime!(E2M1_PER_WORD / run.cells_per_register) {
                    let mut vector = Vector::<T, N>::empty();
                    #[unroll]
                    for e in 0..comptime!(run.cells_per_register) {
                        vector.insert(
                            e,
                            decoded.extract(comptime!(p * run.cells_per_register + e)),
                        );
                    }
                    let cell = comptime!(word + p * run.cells_per_register);
                    fragment[comptime!(run.register(cell, held_row))] = vector;
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

    /// A run two sixteen-deep instructions deep on `map`.
    fn two_steps(
        map: LaneMap,
        ident: MatrixIdent,
        registers_per_step: usize,
        cells_per_register: usize,
    ) -> LaneRun {
        LaneRun {
            ident,
            map,
            depth: 16,
            len: 2 * map.slots_per_unit(16),
            registers_per_step,
            cells_per_register,
            rows: 16,
            k: 32,
        }
    }

    /// Every register of every instruction receives exactly one vector of the run, an
    /// `m16n8k16` `A`'s first pair of `k` in registers 0 (rows `g`) and 1 (rows `g + 8`), an
    /// RDNA3 one's eights in order.
    #[test]
    fn each_register_takes_one_vector_of_the_run() {
        for (map, ident, registers_per_step, cells_per_register) in [
            (LaneMap::M16N8K16, MatrixIdent::A, 4, 2),
            (LaneMap::M16N8K16, MatrixIdent::B, 2, 2),
            (LaneMap::RDNA3_16X16, MatrixIdent::A, 2, 8),
            (LaneMap::RDNA3_16X16, MatrixIdent::B, 2, 8),
        ] {
            let run = two_steps(map, ident, registers_per_step, cells_per_register);
            let mut registers: Vec<_> = (0..run.rows_per_unit())
                .flat_map(|held_row| {
                    (0..run.len)
                        .step_by(run.cells_per_register)
                        .map(move |cell| run.register(cell, held_row))
                })
                .collect();
            registers.sort();
            assert_eq!(
                registers,
                (0..2 * registers_per_step).collect::<Vec<_>>(),
                "{map:?}, {ident:?}"
            );
        }
        let a = two_steps(LaneMap::M16N8K16, MatrixIdent::A, 4, 2);
        assert_eq!(
            (a.register(0, 0), a.register(0, 1), a.register(2, 0)),
            (0, 1, 2)
        );
        let a = two_steps(LaneMap::RDNA3_16X16, MatrixIdent::A, 2, 8);
        assert_eq!(
            (a.register(0, 0), a.register(8, 0), a.register(16, 0)),
            (0, 1, 2)
        );
    }

    /// The units cover sixteen rows of an `A`, and of a `B` eight columns on `m16n8k16`, sixteen
    /// on RDNA3; a window of fewer is short.
    #[test]
    fn a_window_of_fewer_rows_than_the_units_cover_is_short() {
        let run = |map, ident, rows| LaneRun {
            rows,
            ..two_steps(map, ident, 2, 2)
        };
        assert_eq!(
            run(LaneMap::M16N8K16, MatrixIdent::A, 16).covered_rows(),
            16
        );
        assert_eq!(run(LaneMap::M16N8K16, MatrixIdent::B, 8).covered_rows(), 8);
        assert_eq!(
            run(LaneMap::RDNA3_16X16, MatrixIdent::B, 16).covered_rows(),
            16
        );
        assert!(run(LaneMap::RDNA3_16X16, MatrixIdent::B, 8).short());
        assert!(!run(LaneMap::M16N8K16, MatrixIdent::B, 8).short());
    }
}
