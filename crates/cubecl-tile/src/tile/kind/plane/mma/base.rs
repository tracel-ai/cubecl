//! The manual-mma encoding of a plane tile ([`MmaData`]) and its fragment↔memory transports.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2, e4m3,
    prelude::*,
};

use super::block_scaled::*;
use super::widths::*;
use crate::tile::kind::plane::load_matrix::{LDMATRIX_ROW_BYTES, along_k, load_ldmatrix};
use crate::*;

/// One manual-mma fragment: a role's registers plus the shape and transport it dispatches on.
/// `Clone` duplicates the handle, not the registers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct MmaData {
    pub(crate) fragment: MmaFragment,
    #[cube(comptime)]
    pub m: usize,
    #[cube(comptime)]
    pub n: usize,
    #[cube(comptime)]
    pub k: usize,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    #[cube(comptime)]
    pub io: MmaIo,
    /// What the instruction's operands are ([`MmaOperands`]).
    #[cube(comptime)]
    pub operands: MmaOperands,
}

/// How a fragment deeper than its instruction loads, which sets the order its registers hold `k`
/// in. A contraction pairs the registers of its two factors, so both load the same way: lane runs
/// where the contraction loads both out of memory and one out of global memory, a step at a time
/// everywhere else ([`MmaData::mma`]).
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum DeepLoad {
    /// A step at a time, each in the instruction's own order: what `ldmatrix` reads out of shared
    /// memory, and what a fragment loaded on its own, or an accumulator reread, holds
    /// ([`load_in_steps`]).
    Steps,
    /// Each unit's slots one contiguous run of `k`, which it reads in whole loads
    /// ([`load_lane_runs`]).
    LaneRuns,
}

/// One role's register array, each role at its own width (`NL`/`NR`/`NA`), at the element the
/// fragment holds.
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum MmaFragment {
    Lhs(Raw<Registers<NL>>),
    Rhs(Raw<Registers<NR>>),
    Acc(Raw<Registers<NA>>),
    /// A block-scaled `A`: its `e2m1` values as the instruction takes them, two a byte, and the
    /// `e4m3` scales of the row this unit's scale register serves, the one register an array of
    /// one holds, so a handle to the fragment loads the registers the fragment holds.
    LhsBlockScaled(Array<Vector<e2m1x2, NLB>>, Array<Vector<e4m3, NSB>>),
    /// A block-scaled `B`, as [`LhsBlockScaled`](Self::LhsBlockScaled) is an `A`.
    RhsBlockScaled(Array<Vector<e2m1x2, NRB>>, Array<Vector<e4m3, NSB>>),
}

#[cube]
impl MmaData {
    /// Allocate an accumulator fragment.
    pub(crate) fn acc<T: Numeric>(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] operands: MmaOperands,
    ) -> MmaData {
        let depth = instruction_depth::<T>(m, n, k);
        let def = MmaDefinition::<T, T, T>::new(m, n, depth);
        register_acc_size::<T>(&def);
        let count = def.vectors_per_lane(MatrixIdent::Accumulator);
        MmaData {
            fragment: MmaFragment::new_Acc(Raw::<Registers<NA>>::of::<T>(Array::new(count))),
            m,
            n,
            k,
            layout,
            io,
            operands,
        }
    }

    /// Allocate an `A`-role (lhs) operand fragment.
    pub(crate) fn lhs<T: Numeric>(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] operands: MmaOperands,
    ) -> MmaData {
        // A fragment deeper than its instruction holds each of its instructions' registers.
        let depth = instruction_depth::<T>(m, n, k);
        let def = MmaDefinition::<T, T, T>::new(m, n, depth);
        register_lhs_size::<T>(&def);
        let per_step = def.vectors_per_lane(MatrixIdent::A);
        let count = comptime!(per_step * (k / depth));
        MmaData {
            fragment: MmaFragment::new_Lhs(Raw::<Registers<NL>>::of::<T>(Array::new(count))),
            m,
            n,
            k,
            layout,
            io,
            operands,
        }
    }

    /// Allocate a `B`-role (rhs) operand fragment.
    pub(crate) fn rhs<T: Numeric>(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] operands: MmaOperands,
    ) -> MmaData {
        let depth = instruction_depth::<T>(m, n, k);
        let def = MmaDefinition::<T, T, T>::new(m, n, depth);
        register_rhs_size::<T>(&def);
        let per_step = def.vectors_per_lane(MatrixIdent::B);
        let count = comptime!(per_step * (k / depth));
        MmaData {
            fragment: MmaFragment::new_Rhs(Raw::<Registers<NR>>::of::<T>(Array::new(count))),
            m,
            n,
            k,
            layout,
            io,
            operands,
        }
    }

    /// Allocate an operand fragment in role `ident`, of the form `operands` names.
    pub(crate) fn operand<T: Numeric>(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] operands: MmaOperands,
    ) -> MmaData {
        match comptime!((ident, operands)) {
            (_, MmaOperands::BlockScaled) => {
                MmaData::block_scaled(ident, m, n, k, layout, io, operands)
            }
            (MatrixIdent::A, MmaOperands::Plain) => {
                MmaData::lhs::<T>(m, n, k, layout, io, operands)
            }
            (MatrixIdent::B, MmaOperands::Plain) => {
                MmaData::rhs::<T>(m, n, k, layout, io, operands)
            }
            (MatrixIdent::Accumulator, MmaOperands::Plain) => {
                panic!("MmaData::operand: an accumulator is not an operand")
            }
        }
    }

    /// Allocate a block-scaled operand fragment in role `ident`: `e2m1` values under an `e4m3`
    /// scale every [`E2M1_SCALE_BLOCK`] of them along `k`, the device's block-scaled instruction
    /// of `m × n × k` ([`MmaOperands::BlockScaled`]).
    pub(crate) fn block_scaled(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] operands: MmaOperands,
    ) -> MmaData {
        let def = block_scaled_definition::<f32>(m, n, k);
        register_block_scaled_sizes(&def, ident);
        let fragment = match comptime!(ident) {
            MatrixIdent::A => MmaFragment::new_LhsBlockScaled(
                Array::new(def.vectors_per_lane(MatrixIdent::A)),
                Array::new(1usize),
            ),
            MatrixIdent::B => MmaFragment::new_RhsBlockScaled(
                Array::new(def.vectors_per_lane(MatrixIdent::B)),
                Array::new(1usize),
            ),
            MatrixIdent::Accumulator => {
                panic!("MmaData::block_scaled: an accumulator carries no scales")
            }
        };
        MmaData {
            fragment,
            m,
            n,
            k,
            layout,
            io,
            operands,
        }
    }

    /// Zero the fragment, whatever the role.
    pub(crate) fn zero<T: Numeric>(&mut self) {
        match &mut self.fragment {
            MmaFragment::Lhs(f) => fill_registers(f.at_mut::<T>(), T::from_int(0)),
            MmaFragment::Rhs(f) => fill_registers(f.at_mut::<T>(), T::from_int(0)),
            MmaFragment::Acc(f) => fill_registers(f.at_mut::<T>(), T::from_int(0)),
            MmaFragment::LhsBlockScaled(..) | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::zero: a block-scaled operand is loaded, never zeroed")
            }
        }
    }

    /// Multiply every cell of this (accumulator) fragment by `factor`: its registers are the
    /// unit's own cells, so the scale needs no bounce.
    pub(crate) fn scale<T: Numeric>(&mut self, factor: T) {
        match &mut self.fragment {
            MmaFragment::Acc(f) => {
                let f = f.at_mut::<T>();
                let factor = Vector::<T, NA>::cast_from(factor);
                #[unroll]
                for i in 0..f.len() {
                    f[i] *= factor;
                }
            }
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::scale: an operand fragment is contracted, not scaled")
            }
        }
    }

    /// Fill this fragment from `src`'s window by the role's transport, a fragment deeper than its
    /// instruction as `load` says.
    pub(crate) fn load_window<T: Numeric>(&mut self, src: &Tile, #[comptime] load: DeepLoad) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let io = comptime!(self.io);
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        let steps = lane_run_steps::<T>(m, n, k);
        match &mut self.fragment {
            MmaFragment::Lhs(f) => match comptime!(steps > 1) {
                true => load_deep(
                    src,
                    f.at_mut::<T>(),
                    MatrixIdent::A,
                    layout,
                    io,
                    (m, n, k),
                    load,
                ),
                false => load_fragment(
                    src,
                    f.at_mut::<T>(),
                    &def,
                    MatrixIdent::A,
                    layout,
                    io,
                    (m, k),
                    0usize,
                ),
            },
            MmaFragment::Rhs(f) => match comptime!(steps > 1) {
                true => load_deep(
                    src,
                    f.at_mut::<T>(),
                    MatrixIdent::B,
                    layout,
                    io,
                    (m, n, k),
                    load,
                ),
                false => load_fragment(
                    src,
                    f.at_mut::<T>(),
                    &def,
                    MatrixIdent::B,
                    layout,
                    io,
                    (k, n),
                    0usize,
                ),
            },
            MmaFragment::Acc(f) => load_fragment(
                src,
                f.at_mut::<T>(),
                &def,
                MatrixIdent::Accumulator,
                layout,
                io,
                (m, n),
                0usize,
            ),
            MmaFragment::LhsBlockScaled(values, scales) => {
                load_block_scaled(src, values, scales, MatrixIdent::A, layout, io, (m, n, k))
            }
            MmaFragment::RhsBlockScaled(values, scales) => {
                load_block_scaled(src, values, scales, MatrixIdent::B, layout, io, (m, n, k))
            }
        }
    }

    /// Drain this (accumulator) fragment into `mem`'s window, `space` being the window's.
    pub(crate) fn store_window<T: Numeric>(&self, mem: &mut Memory, #[comptime] space: Space) {
        self.store_cast_window::<T, T>(mem, space)
    }

    /// Drain this (accumulator) fragment into `mem`'s window, casting down to the sink element.
    pub(crate) fn store_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: Space,
    ) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        match &self.fragment {
            MmaFragment::Acc(f) => {
                store_cells::<T, Out, T, T, T>(mem, f.at::<T>(), &def, layout, space)
            }
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::store: only an accumulator fragment drains to memory")
            }
        }
    }
}

/// An accumulator read where its units hold it: each register vector runs along one row of the
/// fragment, and a row's cells lie across the units [`MmaDefinition::position_of_nth`] says share
/// it. Slot `first + i` of the arrays these take or return is the row of the unit's register
/// vector `i`, so each unit keeps the state of its own rows and no others.
#[cube]
impl MmaData {
    /// Rows of this accumulator one unit holds cells of, one per register vector.
    pub(crate) fn rows_per_unit<E: Numeric>(&self) -> comptime_type!(usize) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        def.vectors_per_lane(MatrixIdent::Accumulator)
    }

    /// `self[r, c] = self[r, c] · scale + bias[r, c]`, `bias` read at the cell's coordinates in
    /// the window over `space`, this fragment's first cell at `origin` of it.
    pub(crate) fn scale_add_along<E: Numeric>(
        &mut self,
        scale: E,
        bias: &Procedural,
        #[comptime] space: Space,
        #[comptime] origin: (usize, usize),
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let (row0, col0) = comptime!(origin);
        let acc = self.acc_registers_mut::<E>();
        #[unroll]
        for i in 0..acc.len() {
            let mut vector = acc[i];
            #[unroll]
            for e in 0..vector_size {
                let (row, col) = def.position_of_nth(
                    UNIT_POS_PLANE,
                    comptime!((i * vector_size + e) as u32),
                    MatrixIdent::Accumulator,
                );
                let cell = bias.value_at::<E>(
                    row + comptime!(row0 as u32),
                    col + comptime!(col0 as u32),
                    comptime!(space.clone()),
                );
                vector.insert(e, vector.extract(e) * scale + cell);
            }
            acc[i] = vector;
        }
    }

    /// `values[first + r]` written at cell `origin + r` of `window` for every row `r` this unit
    /// holds cells of, by the one unit holding the row's first column: each of the fragment's
    /// rows written once.
    pub(crate) fn store_along<E: Float>(
        &self,
        values: &Array<E>,
        window: &mut Memory,
        #[comptime] first: usize,
        #[comptime] origin: usize,
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let size!(W) = 1usize;
        let mut view = window.flat_mut::<E, W>();
        #[unroll]
        for i in 0..self.rows_per_unit::<E>() {
            let (row, column) = def.position_of_nth(
                UNIT_POS_PLANE,
                comptime!((i * vector_size) as u32),
                MatrixIdent::Accumulator,
            );
            if column == 0 {
                view.write(
                    comptime!(origin) + row as usize,
                    Vector::cast_from(values[first + i]),
                );
            }
        }
    }

    /// `self[r, :] *= factors[first + r]` for every row `r` this unit holds cells of.
    pub(crate) fn mul_along<E: Numeric>(&mut self, factors: &Array<E>, #[comptime] first: usize) {
        let acc = self.acc_registers_mut::<E>();
        #[unroll]
        for i in 0..acc.len() {
            acc[i] *= Vector::cast_from(factors[first + i]);
        }
    }

    /// `slots[first + r]` combined under `monoid` with this unit's cells on its row `r`: the
    /// unit's own part of each row, which [`reduce_across_units`](Self::reduce_across_units)
    /// completes.
    pub(crate) fn reduce_own_cells<E: Numeric>(
        &self,
        slots: &mut Array<E>,
        #[comptime] first: usize,
        #[comptime] monoid: Monoid,
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let acc = self.acc_registers::<E>();
        #[unroll]
        for i in 0..acc.len() {
            let vector = acc[i];
            let mut slot = slots[first + i];
            #[unroll]
            for e in 0..vector_size {
                slot = monoid.combine::<E>(slot, vector.extract(e));
            }
            slots[first + i] = slot;
        }
    }

    /// Each row's slot, from `first` on, combined under `monoid` over the units that share the
    /// row: a butterfly over the lane bits that leave the row in place.
    ///
    /// Whether a lane bit moves a row is read at lane 0, where both positions are constants the
    /// compiler folds, so a bit that moves it costs no shuffle. That holds for a layout whose row
    /// is a function of some of the lane's bits, as the manual-mma layouts are; the register
    /// attention's component test checks it on every lane.
    pub(crate) fn reduce_across_units<E: Numeric>(
        &self,
        slots: &mut Array<E>,
        #[comptime] first: usize,
        #[comptime] monoid: Monoid,
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let rows = def.vectors_per_lane(MatrixIdent::Accumulator);
        let width = mma_plane_width();
        let lane_bits = comptime!(width.trailing_zeros() as usize);
        #[unroll]
        for i in 0..rows {
            let nth = comptime!((i * vector_size) as u32);
            let (row, _) = def.position_of_nth(0u32, nth, MatrixIdent::Accumulator);
            let mut value = slots[first + i];
            #[unroll]
            for bit in 0..lane_bits {
                let mask = comptime!(1u32 << bit);
                let (other_row, _) = def.position_of_nth(mask, nth, MatrixIdent::Accumulator);
                if other_row == row {
                    value = monoid.combine::<E>(value, plane_shuffle_xor(value, mask));
                }
            }
            slots[first + i] = value;
        }
    }

    fn acc_registers<E: Numeric>(&self) -> &Array<Vector<E, NA>> {
        match &self.fragment {
            MmaFragment::Acc(f) => f.at::<E>(),
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData: an operand fragment's cells are contracted, not read by rows")
            }
        }
    }

    fn acc_registers_mut<E: Numeric>(&mut self) -> &mut Array<Vector<E, NA>> {
        match &mut self.fragment {
            MmaFragment::Acc(f) => f.at_mut::<E>(),
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData: an operand fragment's cells are contracted, not read by rows")
            }
        }
    }
}

#[cube]
impl MmaData {
    /// This accumulator with each cell cast to `E` in the register that holds it.
    pub(crate) fn cast<A: Numeric, E: Numeric>(&self) -> MmaData {
        let acc = self.acc_registers::<A>();
        let mut cast = MmaData::acc::<E>(
            comptime!(self.m),
            comptime!(self.n),
            comptime!(self.k),
            comptime!(self.layout),
            comptime!(self.io),
            comptime!(self.operands),
        );
        let registers = cast.acc_registers_mut::<E>();
        #[unroll]
        for i in 0..acc.len() {
            registers[i] = Vector::cast_from(acc[i]);
        }
        cast
    }
}

#[cube]
impl MmaData {
    /// Each row's max, starting from `seed`'s, over the units that share it.
    pub(crate) fn maxima_along<E: Float>(&self, seed: &Array<E>) -> Array<E> {
        let rows = self.rows_per_unit::<E>();
        let mut maxima = Array::<E>::new(rows);
        #[unroll]
        for i in 0..rows {
            maxima[i] = seed[i];
        }
        self.reduce_own_cells(&mut maxima, 0usize, Monoid::Max);
        self.reduce_across_units(&mut maxima, 0usize, Monoid::Max);
        maxima
    }

    /// Each row's sum over the units that share it.
    pub(crate) fn sums_along<E: Float>(&self) -> Array<E> {
        let rows = self.rows_per_unit::<E>();
        let mut sums = Array::<E>::new(rows);
        #[unroll]
        for i in 0..rows {
            sums[i] = E::from_int(0);
        }
        self.reduce_own_cells(&mut sums, 0usize, Monoid::Sum);
        self.reduce_across_units(&mut sums, 0usize, Monoid::Sum);
        sums
    }

    /// `self[r, c] = exp(self[r, c] − slices[first + r])` ([`AxisSlices::exp_minus_cell`]).
    pub(crate) fn exp_minus_along<E: Float>(
        &mut self,
        slices: &Array<E>,
        #[comptime] first: usize,
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let acc = self.acc_registers_mut::<E>();
        #[unroll]
        for i in 0..acc.len() {
            let mut vector = acc[i];
            #[unroll]
            for e in 0..vector_size {
                vector.insert(
                    e,
                    AxisSlices::exp_minus_cell::<E>(vector.extract(e), slices[first + i]),
                );
            }
            acc[i] = vector;
        }
    }
}

/// The plane width the manual-mma layouts are stated for, read off the target at expansion.
#[cube]
fn mma_plane_width() -> comptime_type!(u32) {
    intrinsic!(|scope| scope.state().target_properties.mma.const_plane_size)
}

#[cube]
fn register_acc_size<A: Numeric>(def: &MmaDefinition<A, A, A>) {
    let va = def.vector_size(MatrixIdent::Accumulator);
    intrinsic!(|scope| {
        scope.register_size::<NA>(va);
    });
}

#[cube]
fn register_lhs_size<L: Numeric>(def: &MmaDefinition<L, L, L>) {
    let vl = def.vector_size(MatrixIdent::A);
    intrinsic!(|scope| {
        scope.register_size::<NL>(vl);
    });
}

#[cube]
fn register_rhs_size<R: Numeric>(def: &MmaDefinition<R, R, R>) {
    let vr = def.vector_size(MatrixIdent::B);
    intrinsic!(|scope| {
        scope.register_size::<NR>(vr);
    });
}

/// Fill every register slot with `value`.
#[cube]
fn fill_registers<E: Numeric, N: Size>(fragment: &mut Array<Vector<E, N>>, value: E) {
    let num_vectors = fragment.len();
    let v = Vector::<E, N>::cast_from(value);
    #[unroll]
    for i in 0..num_vectors {
        fragment[i] = v;
    }
}

/// Load `fragment` (role `ident`, shaped by `edges`) from a row-major window.
#[cube]
pub(crate) fn load_fragment<T: Numeric, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] edges: (usize, usize),
    #[comptime] k_offset: usize,
) {
    // `ldmatrix` reads 16-byte rows of 16-bit served cells from shared memory, on a device that
    // offers it for the element; any other window, or device, takes the manual load.
    let offered = loads_matrix::<T>();
    let gathered = src.gathered();
    let shared = src.is_shared();
    let element = src.stage_element();
    let holds_served_values = comptime!(element.holds_values());
    let served = src.vector_size();
    // An element's size read at expansion, where the launch has registered it: inside
    // `comptime!` the call would size the generic placeholder instead.
    let elem_size = T::size().comptime();
    let row_cells = comptime!(LDMATRIX_ROW_BYTES / elem_size);
    // `ldmatrix` reads every row of its matrices: a window covering fewer has none to give it.
    let whole = comptime!(covered_edges(&src.place.space, ident, edges) == edges);
    let ldmatrix_serves = comptime!(
        offered
            && shared
            && !gathered
            && holds_served_values
            && elem_size == 2
            && ident != MatrixIdent::Accumulator
            && row_cells.is_multiple_of(served)
            && whole
    );
    let method = comptime!(match ldmatrix_serves {
        true => io.load_method(ident),
        false => LoadMethod::Manual,
    });
    match method {
        LoadMethod::Manual => {
            let size!(W) = src.vector_size();
            load_manual::<T, W, N, A, B, CD>(src, fragment, def, ident, layout, edges, k_offset)
        }
        LoadMethod::LoadMatrix => {
            let size!(W) = src.vector_size();
            load_ldmatrix::<T, W, N, A, B, CD>(src, fragment, def, ident, layout, edges, k_offset)
        }
    }
}

/// Whether the device offers `ldmatrix` for `T`, read off its properties at expansion.
// `T` is read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
fn loads_matrix<T: Numeric>() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let element = T::elem_type(scope);
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| properties.features.matmul.ldmatrix.contains(&element))
    })
}

/// A fragment deeper than its instruction, out of `src`, as `load` says ([`load_in_steps`],
/// [`load_lane_runs`]).
#[cube]
fn load_deep<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] shape: (usize, usize, usize),
    #[comptime] load: DeepLoad,
) {
    match comptime!(load) {
        DeepLoad::Steps => load_in_steps(src, fragment, ident, layout, io, shape),
        DeepLoad::LaneRuns => load_lane_runs(src, fragment, ident, layout, shape),
    }
}

/// Manual load: each unit reads its own cells of `src` through the matrix view, `k_offset` along
/// `k` into a window deeper than one instruction.
#[cube]
fn load_manual<T: Numeric, W: Size, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] edges: (usize, usize),
    #[comptime] k_offset: usize,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let vector_layout = def.vector_layout(ident);
    let unit_id = UNIT_POS_PLANE;
    let served = src.vector_size();
    let width = comptime!(served);
    // A window may cover fewer of the fragment's rows (an `A`) or columns (a `B`) than the
    // fragment does, a decode's rows staged as they are: the units on cells past it read zeros
    // rather than past its buffer.
    let (rows, cols) = comptime!(covered_edges(&src.place.space, ident, edges));
    let short = comptime!((rows, cols) != edges);
    let transposed = comptime!(match layout {
        MatrixLayout::RowMajor => false,
        MatrixLayout::ColMajor => true,
        MatrixLayout::Undefined => {
            panic!("MmaData::load: a manual fragment load reads a row- or col-major window")
        }
    });
    // The window's edge along its lines must hold whole lines, or a line index rounds down.
    let line_edge = comptime!(if transposed { rows } else { cols });
    comptime!(assert!(
        line_edge.is_multiple_of(width),
        "MmaData::load: a {width}-wide line runs past the fragment's {line_edge}-cell edge along \
         it, which a manual fragment load cannot read from inside the line; serve the operand at \
         most {line_edge} wide"
    ));
    let view = if comptime!(transposed) {
        src.fragment_matrix_packed::<T, W>(cols, rows)
    } else {
        src.fragment_matrix_packed::<T, W>(rows, cols)
    };
    let along_lines = comptime!(vector_layout == layout);
    comptime!(assert!(
        !along_lines || vector_size.is_multiple_of(width) || width.is_multiple_of(vector_size),
        "MmaData::load: a {vector_size}-cell register vector neither holds whole {width}-wide \
         lines nor sits inside one"
    ));

    #[unroll]
    for i in 0..num_vectors {
        let mut vector = Vector::empty();
        if comptime!(along_lines) {
            let (row, col) = def.position_of_nth(unit_id, comptime!(i * vector_size) as u32, ident);
            let (row, col) = along_k(row, col, ident, k_offset);
            let (line_row, cell) = if comptime!(transposed) {
                (col, row)
            } else {
                (row, col)
            };
            // A register vector lies along the lines, so its cells share the row or column a
            // short window cuts.
            let covered = !short || (row < comptime!(rows as u32) && col < comptime!(cols as u32));
            if !covered {
                #[unroll]
                for e in 0..vector_size {
                    vector.insert(e, T::from_int(0));
                }
            } else if comptime!(width >= vector_size) {
                // A register vector starts at a multiple of its size, so it stays within `width`.
                let line = view.read((line_row, cell / comptime!(width as u32)));
                let start = cell % comptime!(width as u32);
                #[unroll]
                for e in 0..vector_size {
                    vector.insert(e, line.extract_dynamic((start + e as u32).cast::<usize>()));
                }
            } else {
                #[unroll]
                for l in 0..comptime!(vector_size / width) {
                    let line = view.read((
                        line_row,
                        cell / comptime!(width as u32) + comptime!(l as u32),
                    ));
                    #[unroll]
                    for e in 0..width {
                        vector.insert(comptime!(l * width + e), line.extract(e));
                    }
                }
            }
        } else {
            #[unroll]
            for e in 0..vector_size {
                let elem_idx = i * vector_size + e;
                let (row, col) = def.position_of_nth(unit_id, elem_idx as u32, ident);
                let (row, col) = along_k(row, col, ident, k_offset);
                let (line_row, cell) = if comptime!(transposed) {
                    (col, row)
                } else {
                    (row, col)
                };
                vector.insert(e, T::from_int(0));
                if !short || (row < comptime!(rows as u32) && col < comptime!(cols as u32)) {
                    let line = view.read((line_row, cell / comptime!(width as u32)));
                    vector.insert(
                        e,
                        line.extract_dynamic((cell % comptime!(width as u32)).cast::<usize>()),
                    );
                }
            }
        }
        fragment[i] = vector;
    }
}

/// The `rows x cols` of a fragment's `edges` that `space`, the window it loads from, covers: all
/// of them, or, where the window is shorter than the fragment off `k`, the rows of an `A` or the
/// columns of a `B` it has.
fn covered_edges(space: &Space, ident: MatrixIdent, edges: (usize, usize)) -> (usize, usize) {
    let cells: usize = space.axes().map(|axis| space.extent(axis)).product();
    let (rows, cols) = edges;
    match ident {
        MatrixIdent::A if cells.is_multiple_of(cols) && cells / cols < rows => (cells / cols, cols),
        MatrixIdent::B if cells.is_multiple_of(rows) && cells / rows < cols => (rows, cells / rows),
        MatrixIdent::A | MatrixIdent::B | MatrixIdent::Accumulator => edges,
    }
}

/// Each unit's accumulator cells written one element at a time through `mem`'s own write.
#[cube]
fn store_cells<T: Numeric, Out: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &mut Memory,
    fragment: &Array<Vector<T, NA>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] layout: MatrixLayout,
    #[comptime] space: Space,
) {
    comptime!(assert!(
        mem.store.vector_size == 1,
        "MmaData: a fragment drained cell by cell writes one element at a time, and its \
         destination is bound {} wide; bind it one element wide",
        mem.store.vector_size
    ));
    let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);
    let vector_size = def.vector_size(MatrixIdent::Accumulator);
    let unit_id = UNIT_POS_PLANE;
    let axes = comptime!(MatrixAxes::trailing(&space));
    let mut sink = mem.matrix_mut::<Out, Const<1>>(0usize, axes, space);

    #[unroll]
    for i in 0..num_vectors {
        #[unroll]
        for e in 0..vector_size {
            let elem_idx = i * vector_size + e;
            let (row, col) =
                def.position_of_nth(unit_id, elem_idx as u32, MatrixIdent::Accumulator);
            let at = match comptime!(layout) {
                MatrixLayout::RowMajor => (row, col),
                MatrixLayout::ColMajor => (col, row),
                MatrixLayout::Undefined => panic!("mma: a stage layout must be row- or col-major"),
            };
            let mut value = Vector::<Out, Const<1>>::empty();
            value.insert(0usize, Out::cast_from(fragment[i].extract(e)));
            sink.write(at, value);
        }
    }
}

/// An `m × k` accumulator's registers regrouped as the `A` operand of an `m × n × k` contraction:
/// for 16-bit operands the manual-mma layouts put the accumulator's cell `e` where the operand's
/// cell `e` is, so a product contracting its own output reads it without moving a cell between
/// units. A wider operand's `A` lays its cells otherwise (a `tf32` one along the columns first),
/// and is refused.
#[cube]
pub(crate) fn accumulator_as_a<L: Numeric>(
    acc: &Array<Vector<L, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> Array<Vector<L, NL>> {
    let depth = instruction_depth::<L>(m, n, k);
    comptime!(assert!(
        depth == k,
        "accumulator_as_a: an accumulator read as the A operand is one instruction deep; a \
         {k}-deep contraction of it would hold `k` in lane runs, which its cells do not lie in"
    ));
    let acc_def = MmaDefinition::<L, L, L>::new(m, k, k);
    let a_def = MmaDefinition::<L, L, L>::new(m, n, k);
    register_lhs_size::<L>(&a_def);
    let acc_regs = acc_def.vectors_per_lane(MatrixIdent::Accumulator);
    let acc_width = acc_def.vector_size(MatrixIdent::Accumulator);
    let a_regs = a_def.vectors_per_lane(MatrixIdent::A);
    let a_width = a_def.vector_size(MatrixIdent::A);
    let bytes = L::size().comptime();
    comptime!(assert!(
        bytes == 2,
        "accumulator_as_a: an accumulator's cells sit where the A operand's do for 16-bit \
         operands; a {bytes}-byte operand's A lays them otherwise"
    ));
    comptime!(assert!(
        acc_regs * acc_width == a_regs * a_width,
        "accumulator_as_a: an {m}x{k} accumulator holds {} cells a unit, and the A operand of an \
         {m}x{n}x{k} contraction {}",
        acc_regs * acc_width,
        a_regs * a_width
    ));
    let mut registers = Array::<Vector<L, NL>>::new(a_regs);
    #[unroll]
    for i in 0..acc_regs {
        #[unroll]
        for e in 0..acc_width {
            let (register, slot) =
                comptime!(((i * acc_width + e) / a_width, (i * acc_width + e) % a_width));
            let mut vector = registers[register];
            vector.insert(slot, acc[i].extract(e));
            registers[register] = vector;
        }
    }
    registers
}

/// `acc += lhs · rhs` over three role fragments via `MmaDefinition::execute`: one instruction,
/// or, where the fragments are deeper than it, each of theirs in turn ([`lane_runs`]).
#[cube]
pub(crate) fn mma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let depth = instruction_depth::<L>(m, n, k);
    if comptime!(depth < k) {
        let def = MmaDefinition::<L, R, A>::new(m, n, depth);
        mma_execute_lane_runs::<L, R, A>(&def, lhs, rhs, acc, comptime!(k / depth));
    } else {
        let def = MmaDefinition::<L, R, A>::new(m, n, k);
        let out = def.execute(lhs, rhs, &*acc);
        let num = def.vectors_per_lane(MatrixIdent::Accumulator);
        #[unroll]
        for i in 0..num {
            acc[i] = out[i];
        }
    }
}
