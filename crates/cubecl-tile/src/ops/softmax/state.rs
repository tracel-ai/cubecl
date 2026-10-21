//! The fold's running state and the masking probe.

use cubecl::prelude::*;

use crate::*;

/// Logits at or below this are treated as masked (effectively -inf). Fits f16.
pub const LOGIT_MASKED: f64 = -6e4;

/// Below this an `l` row sum is numerically zero (fully-masked row).
pub(crate) const FULLY_MASKED_ROW_THRESHOLD: f64 = 1e-4;

/// `1/l`, exactly zero when `l` is numerically zero.
#[cube]
pub fn masked_recip<E: Float>(l: E) -> E {
    let eps = E::new(FULLY_MASKED_ROW_THRESHOLD);
    E::cast_from(l >= eps) * clamp_min(l, eps).recip()
}

/// How the score rows are shared out, and therefore how a row reduction closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowShare {
    /// One unit per row-slice: the reduction runs in its own registers.
    Unit { rows: usize },
    /// One plane per row-slice: its units split the reduced axis and meet in a plane reduction.
    /// The cube's x dim must be whole planes and `units` the device's committed width.
    Plane { rows: usize, units: usize },
}

impl RowShare {
    /// Rows one worker owns.
    pub fn rows(&self) -> usize {
        match self {
            RowShare::Unit { rows } | RowShare::Plane { rows, .. } => *rows,
        }
    }

    /// Units one worker spans: one, or the plane's width.
    pub fn units(&self) -> usize {
        match self {
            RowShare::Unit { .. } => 1,
            RowShare::Plane {
                units: plane_units, ..
            } => *plane_units,
        }
    }
}

/// This unit's unit within its worker: its position in the plane, or zero for a unit.
#[cube]
pub(crate) fn owned_unit(#[comptime] share: RowShare) -> usize {
    match comptime!(share) {
        RowShare::Unit { rows: _ } => 0usize,
        RowShare::Plane {
            rows: _,
            units: plane_units,
        } => UNIT_POS_X as usize % plane_units,
    }
}

/// Per-row running state `(m, l)` of the online softmax, in the owning worker's registers.
#[derive(CubeType)]
pub struct RowState<E: Float> {
    pub m: Array<E>,
    pub l: Array<E>,
    #[cube(comptime)]
    pub space: Space,
    /// Who owns which rows.
    #[cube(comptime)]
    pub share: RowShare,
    /// This unit's place in the team sharing the tile. Unread under [`RowShare::Plane`].
    pub team: TeamUnit,
}

#[cube]
impl<E: Float> RowState<E> {
    /// `space` is the kept axes; unit u owns rows `[u*rpu, (u+1)*rpu)` of `units`.
    pub fn new(#[comptime] space: Space, #[comptime] units: usize) -> RowState<E> {
        let rows = comptime!(space.cells().div_ceil(units));
        RowState::<E>::of(space, comptime!(RowShare::Unit { rows }))
    }

    /// [`new`](RowState::new) at plane ownership, its units splitting each row's reduced axis.
    /// `units` must be the device's committed plane width.
    pub fn over_plane(#[comptime] space: Space, #[comptime] plane_units: usize) -> RowState<E> {
        let rows = comptime!(space.cells());
        RowState::<E>::of(
            space,
            comptime!(RowShare::Plane {
                rows,
                units: plane_units
            }),
        )
    }

    /// The state one worker holds, its team laid along the cube's x dim.
    pub fn of(#[comptime] space: Space, #[comptime] share: RowShare) -> RowState<E> {
        RowState::<E>::in_team(space, share, &TeamUnit::along_x())
    }

    /// [`of`](RowState::of) for a worker whose place in its team the caller states.
    pub fn in_team(
        #[comptime] space: Space,
        #[comptime] share: RowShare,
        team: &TeamUnit,
    ) -> RowState<E> {
        let rows = comptime!(share.rows());
        let mut m = Array::new(rows);
        let mut l = Array::new(rows);
        for i in 0..rows {
            m[i] = E::min_value();
            l[i] = E::from_int(0);
        }
        RowState::<E> {
            m,
            l,
            space,
            share,
            team: team.clone(),
        }
    }

    /// The row of the tile this worker's `ri`-th owned row is.
    pub(crate) fn owned_row(&self, ri: usize) -> usize {
        match comptime!(self.share) {
            RowShare::Unit { rows } => self.team.index * rows + ri,
            RowShare::Plane { rows: _, units: _ } => ri,
        }
    }

    /// Absorb one block's row maxes and sums; returns `corr = exp(m_old - m_new)` per row.
    pub fn update(&mut self, max_buf: &Array<E>, sum_buf: &Array<E>) -> Array<E> {
        let rows = comptime!(self.share.rows());
        let mut corr = Array::new(rows);
        for i in 0..rows {
            corr[i] = (self.m[i] - max_buf[i]).exp();
            self.l[i] = corr[i] * self.l[i] + sum_buf[i];
            self.m[i] = max_buf[i];
        }
        corr
    }

    /// Epilogue `lse = m + ln(l)`. Fully-masked rows give -inf via `ln(0)`.
    pub fn lse(&self, i: usize) -> E {
        self.m[i] + E::ln(self.l[i])
    }

    /// Epilogue `1/l` for row `i`, with the fully-masked guard.
    pub fn recip_l(&self, i: usize) -> E {
        masked_recip::<E>(self.l[i])
    }
}

/// The masking predicates and where the score tile sits in the global (kept, reduced) space.
/// Row `r` sits at query `origin_q + r % q_rows`.
#[derive(CubeType)]
pub struct MaskProbe {
    pub origin_q: usize,
    /// The score row the probed tile's first row is, in the rows `q_rows` maps to positions.
    pub row_origin: usize,
    pub origin_s: usize,
    pub bound_q: usize,
    pub bound_s: usize,
    #[cube(comptime)]
    pub q_rows: usize,
    #[cube(comptime)]
    pub causal: bool,
    #[cube(comptime)]
    pub materialized: bool,
}

#[cube]
impl MaskProbe {
    /// Bound, causal, and materialized predicates at global position (q, s).
    pub fn masked(&self, q: usize, s: usize, mask: &Tile) -> bool {
        let mut masked = q >= self.bound_q || s >= self.bound_s;
        if comptime!(self.causal) {
            masked = masked || s > q;
        }
        if comptime!(self.materialized) {
            let size!(W) = mask.vector_size();
            let rank = comptime!(mask.place.space.rank());
            let cols = mask.runtime_extent(comptime!(mask.place.space.axis_at(rank - 1)));
            masked = masked || mask.flat::<u32, W>().read(q * cols + s).extract(0usize) != 0;
        }
        masked
    }

    /// How many key positions this tile's rows can read; every score past it is masked.
    pub fn keys(&self) -> usize {
        let mut keys = self.bound_s;
        if comptime!(self.causal) {
            keys = keys.min_with(self.origin_q.plus(comptime!(self.q_rows).runtime()));
        }
        keys
    }

    /// [`keys`](MaskProbe::keys) counted in whole blocks of `block`, the partial last included.
    pub fn blocks(&self, #[comptime] block: usize) -> usize {
        self.keys().div_ceil(block)
    }

    /// The query position of score row `r` (see `q_rows`).
    pub(crate) fn row_q(&self, r: usize) -> usize {
        let q_rows = comptime!(self.q_rows);
        self.origin_q + (self.row_origin + r) % q_rows
    }

    /// The probe advanced `offset` along the reduced axis.
    pub fn step_s(&self, offset: usize) -> MaskProbe {
        MaskProbe {
            origin_q: self.origin_q,
            row_origin: self.row_origin,
            origin_s: self.origin_s + offset,
            bound_q: self.bound_q,
            bound_s: self.bound_s,
            q_rows: comptime!(self.q_rows),
            causal: comptime!(self.causal),
            materialized: comptime!(self.materialized),
        }
    }
}
