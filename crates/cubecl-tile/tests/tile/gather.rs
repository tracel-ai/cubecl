//! The memory gather leaf with a vectorized lhs, against a host reference that reads every operand
//! through its projection: vectorized along the rows (a convolution's weight gradient), along the
//! contraction under a backward window (its data gradient), and a forward convolution over two
//! tap axes.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::prelude::*;

use super::conv::{conv_kernel, ramp};
use super::{Form, implied};
use cubecl_tile::layout::{PhysicalAxisMap, Scale};
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};

const RH: Axis = Axis(0);
const CI: Axis = Axis(1);
const CO: Axis = Axis(2);
const B: Axis = Axis(3);
const OH: Axis = Axis(4);
const OW: Axis = Axis(5);
const RW: Axis = Axis(6);

const BLOCK: RegisterBlock = RegisterBlock::new(64);
const ROLLED: RegisterBlock = RegisterBlock::new(1);
const SPLIT: RegisterBlock = BLOCK.split_edge();

/// One operand: its buffer's shape, its spec, and the vector width it is served at.
struct Operand {
    shape: Vec<usize>,
    spec: TileSpec,
    width: usize,
}

impl Operand {
    /// An operand addressed directly over `axes` of `space`, masked as `policy` says.
    fn direct(
        space: &[(Axis, usize)],
        axes: &[Axis],
        width: usize,
        policy: BoundaryPolicy,
    ) -> Self {
        let extent = |axis| space.iter().find(|(a, _)| *a == axis).unwrap().1;
        Operand {
            shape: axes.iter().map(|&a| extent(a)).collect(),
            spec: TileSpec::direct(axes).boundary(policy),
            width,
        }
    }

    /// The flat index `coord`, indexed by axis, reads; `None` past the buffer.
    fn index(&self, coord: &[usize]) -> Option<usize> {
        let p = &self.spec.projection;
        (0..p.physical_rank()).try_fold(0, |flat, pa| {
            let terms = self.spec.axes().iter();
            let at = p.offset(pa).get()
                + terms
                    .map(|&a| p.scale(pa, a) * coord[a.0 as usize] as isize)
                    .sum::<isize>();
            let len = self.shape[pa];
            (0..len as isize)
                .contains(&at)
                .then(|| flat * len + at as usize)
        })
    }

    fn launch(&self, client: &Client, data: Vec<f32>) -> TileTensorLaunch {
        let (handle, _) = TestInput::builder(client.clone(), self.shape.clone())
            .dtype(f32::elem_type_native())
            .custom(data)
            .generate_with_f32_host_data();
        TileTensorLaunch::new(
            handle.binding().into_tensor_arg(),
            f32::elem_type_native(),
            self.width,
            self.spec.clone(),
        )
    }
}

/// `out = Σ lhs · rhs` over the axes of `space` the output lacks, walked in leaf tiles of `tile`.
struct Gather {
    space: Vec<(Axis, usize)>,
    tile: Vec<(Axis, usize)>,
    lhs: Operand,
    rhs: Operand,
    out_axes: Vec<Axis>,
}

impl Gather {
    fn check(&self, config: RegisterBlock, case: &dyn std::fmt::Debug) {
        let client = cubecl::test_device().client();
        let axes: Vec<Axis> = self.space.iter().map(|(a, _)| *a).collect();
        let launcher = implied(
            &client,
            Partitioning::new(
                Space::new(&self.space),
                Levels::leaf(&self.tile).walk_every(&axes).build(),
            ),
            Form::Static,
        );
        let out = Operand::direct(&self.space, &self.out_axes, 1, checked());
        let x = ramp(self.lhs.shape.iter().product(), 7);
        let w = ramp(self.rhs.shape.iter().product(), 5);
        let out_handle = TestInput::builder(client.clone(), out.shape.clone())
            .dtype(f32::elem_type_native())
            .zeros()
            .generate_without_host_data();
        conv_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            self.lhs.launch(&client, x.clone()),
            self.rhs.launch(&client, w.clone()),
            TileTensorLaunch::new(
                out_handle.clone().binding().into_tensor_arg(),
                f32::elem_type_native(),
                1,
                out.spec.clone(),
            ),
            config,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        );
        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);

        let mut want = vec![0.0f32; out.shape.iter().product()];
        let extents: Vec<usize> = self.space.iter().map(|(_, n)| *n).collect();
        for flat in 0..extents.iter().product() {
            let mut coord = [0usize; 8];
            for (a, c) in axes.iter().zip(unravel(flat, &extents)) {
                coord[a.0 as usize] = c;
            }
            if let (Some(i), Some(j)) = (self.lhs.index(&coord), self.rhs.index(&coord)) {
                want[out.index(&coord).unwrap()] += x[i] * w[j];
            }
        }
        for (flat, want) in want.iter().enumerate() {
            let at = unravel(flat, &out.shape);
            assert_eq!(
                got.get_f32(&at),
                *want,
                "{case:?} under {config:?}: wrong at {at:?}"
            );
        }
    }
}

/// `flat`'s row-major coordinate over `extents`.
fn unravel(mut flat: usize, extents: &[usize]) -> Vec<usize> {
    let mut at = vec![0; extents.len()];
    for (p, &n) in extents.iter().enumerate().rev() {
        at[p] = flat % n;
        flat /= n;
    }
    at
}

fn checked() -> BoundaryPolicy {
    BoundaryPolicy::Every(Boundary::Zero)
}

/// A convolution window along one spatial axis.
#[derive(Clone, Copy, Debug)]
struct Window {
    stride: usize,
    dilation: usize,
    padding: usize,
}

const DENSE: Window = Window {
    stride: 1,
    dilation: 1,
    padding: 0,
};
const PADDED: Window = Window {
    padding: 1,
    ..DENSE
};
/// Padding wider than a tap run, so whole runs fall outside the input.
const WIDE: Window = Window {
    padding: 2,
    ..DENSE
};
const SPACED: Window = Window {
    stride: 2,
    dilation: 2,
    padding: 1,
};

impl Window {
    /// The input position `out · stride + tap · dilation - padding`.
    fn map(self, out: Axis, tap: Axis) -> PhysicalAxisMap {
        PhysicalAxisMap::affine(&[(out, self.stride), (tap, self.dilation)])
            .shifted(-(self.padding as isize))
    }

    /// The input length `outputs` positions of `taps` taps reach, padded on both sides.
    fn input(self, outputs: usize, taps: usize) -> usize {
        (outputs - 1) * self.stride + (taps - 1) * self.dilation + 1 - 2 * self.padding
    }
}

/// A convolution's weight gradient: the input vectorized along `CI`, an accumulator row axis, the
/// output gradient vectorized along `CO`, `B` and `OH` contracted and `RH` a batch axis.
#[derive(Clone, Copy, Debug)]
struct WeightBackward {
    ci: usize,
    co: usize,
    window: Window,
    widths: VectorWidths,
}

/// The input's and the output gradient's vector widths.
#[derive(Clone, Copy, Debug)]
struct VectorWidths {
    input: usize,
    grad: usize,
}

impl WeightBackward {
    const PLAIN: Self = Self {
        ci: 8,
        co: 4,
        window: DENSE,
        widths: VectorWidths { input: 4, grad: 2 },
    };

    fn check(self, config: RegisterBlock) {
        let (b, oh, rh) = (2, 5, 3);
        let space = vec![(RH, rh), (CI, self.ci), (CO, self.co), (B, b), (OH, oh)];
        let lhs = Operand {
            shape: vec![b, self.window.input(oh, rh), self.ci],
            spec: TileSpec::new(Projection::new(
                &[B, OH, RH, CI],
                &[
                    PhysicalAxisMap::of(B),
                    self.window.map(OH, RH),
                    PhysicalAxisMap::of(CI),
                ],
            ))
            .boundary(checked()),
            width: self.widths.input,
        };
        Gather {
            tile: vec![(RH, rh), (CI, 8), (CO, 4), (B, b), (OH, oh)],
            rhs: Operand::direct(&space, &[B, OH, CO], self.widths.grad, checked()),
            lhs,
            out_axes: vec![RH, CI, CO],
            space,
        }
        .check(config, &self);
    }
}

/// Plain, split at the edge, with tiles overhanging both extents under a spaced window, rolled,
/// and at each vector width pairing the gradient serves.
#[test]
fn gather_rows_vectorized_lhs() {
    let overhang = WeightBackward {
        ci: 12,
        co: 6,
        window: SPACED,
        ..WeightBackward::PLAIN
    };
    let spaced = WeightBackward {
        window: SPACED,
        ..WeightBackward::PLAIN
    };
    let widths = |input, grad| WeightBackward {
        widths: VectorWidths { input, grad },
        ..WeightBackward::PLAIN
    };
    for (case, config) in [
        (WeightBackward::PLAIN, BLOCK),
        (WeightBackward::PLAIN, SPLIT),
        (overhang, BLOCK),
        (spaced, ROLLED),
        (widths(2, 1), BLOCK),
        (widths(4, 4), BLOCK),
    ] {
        case.check(config);
    }
}

/// A convolution's data gradient: `grad[o + 1 - r]` gathered under a backward window scale,
/// vectorized along `CI`, which is contracted together with `RH`.
#[derive(Clone, Copy, Debug)]
struct DataBackward {
    oh: usize,
    grad_len: usize,
    width: usize,
    tile_oh: usize,
}

impl DataBackward {
    fn check(self, config: RegisterBlock) {
        let (co, rh, ci) = (4, 3, 4);
        let space = vec![(OH, self.oh), (CO, co), (RH, rh), (CI, ci)];
        let lhs = Operand {
            shape: vec![self.grad_len, ci],
            spec: TileSpec::new(Projection::new(
                &[OH, RH, CI],
                &[
                    PhysicalAxisMap::scaled(&[(OH, Scale::Static(1)), (RH, Scale::Static(-1))])
                        .shifted(1),
                    PhysicalAxisMap::of(CI),
                ],
            ))
            .boundary(checked()),
            width: self.width,
        };
        Gather {
            tile: vec![(OH, self.tile_oh), (CO, co), (RH, rh), (CI, ci)],
            rhs: Operand::direct(&space, &[RH, CI, CO], 1, BoundaryPolicy::Unchecked),
            lhs,
            out_axes: vec![OH, CO],
            space,
        }
        .check(config, &self);
    }
}

/// A backward window's reads stay masked at every tile, whatever the gradient's length and the
/// lhs width: `guard_provable` refuses a backward coefficient, so no tile retires its masks.
#[test]
fn gather_contraction_vectorized_lhs_backward_window() {
    let case = |oh, grad_len, width, tile_oh| DataBackward {
        oh,
        grad_len,
        width,
        tile_oh,
    };
    for (case, config) in [
        (case(8, 7, 4, 8), BLOCK),
        (case(8, 7, 2, 3), ROLLED),
        (case(12, 11, 4, 2), SPLIT),
        (case(12, 4, 4, 3), SPLIT),
        (case(8, 7, 1, 8), SPLIT),
    ] {
        case.check(config);
    }
}

/// A forward convolution over two spatial axes: the input vectorized along `CI` and contracted with
/// the taps `RH` and `RW`, so the walk has two outer tap axes and one fastest one.
#[derive(Clone, Copy, Debug)]
struct Forward {
    ci: usize,
    co: usize,
    width: usize,
    window: Window,
    /// Leaf tile edges along `OH`, `OW` and `CO`, past which the outputs overhang.
    tile: (usize, usize, usize),
    /// Whether the input also spans `CO`, so each cell reads its own vector.
    per_cell: bool,
}

impl Forward {
    const PLAIN: Self = Self {
        ci: 8,
        co: 4,
        width: 4,
        window: DENSE,
        tile: (5, 5, 4),
        per_cell: false,
    };

    fn check(self, config: RegisterBlock) {
        let (b, o, taps, co) = (2, 5, 3, self.co);
        let len = self.window.input(o, taps);
        let (th, tw, tco) = self.tile;
        let tile = |oh, ow, co| {
            vec![
                (B, b),
                (OH, oh),
                (OW, ow),
                (CO, co),
                (RH, taps),
                (RW, taps),
                (CI, self.ci),
            ]
        };
        let space = tile(o, o, co);
        let mut axes = vec![B, OH, OW, RH, RW, CI];
        let mut maps = vec![
            PhysicalAxisMap::of(B),
            self.window.map(OH, RH),
            self.window.map(OW, RW),
            PhysicalAxisMap::of(CI),
        ];
        let mut shape = vec![b, len, len, self.ci];
        if self.per_cell {
            axes.insert(3, CO);
            maps.insert(3, PhysicalAxisMap::of(CO));
            shape.insert(3, co);
        }
        let lhs = Operand {
            shape,
            spec: TileSpec::new(Projection::new(&axes, &maps)).boundary(checked()),
            width: self.width,
        };
        let rhs = Operand::direct(&space, &[RH, RW, CI, CO], 1, BoundaryPolicy::Unchecked);
        Gather {
            tile: tile(th, tw, tco),
            rhs,
            lhs,
            out_axes: vec![B, OH, OW, CO],
            space,
        }
        .check(config, &self);
    }
}

/// Anchored runs crossing or wholly past a padded edge, scalar widths on the flat walk, the step
/// loops rolled, tiles with an interior that skip the masks, tiles overhanging the output
/// channels, and an input read per cell with and without the component fan-out.
#[test]
fn gather_tap_walk_forward() {
    let case = |width, window, tile| Forward {
        width,
        window,
        tile,
        ..Forward::PLAIN
    };
    let per_cell = Forward {
        per_cell: true,
        ..Forward::PLAIN
    };
    let overhang = Forward {
        co: 6,
        ..Forward::PLAIN
    };
    for (case, config) in [
        (Forward::PLAIN, BLOCK),
        (case(4, SPACED, (3, 2, 4)), BLOCK),
        (case(2, WIDE, (2, 2, 4)), SPLIT),
        (case(1, PADDED, (5, 5, 4)), BLOCK),
        (case(2, PADDED, (3, 3, 4)), ROLLED),
        (case(4, PADDED, (1, 1, 4)), SPLIT),
        (overhang, BLOCK),
        (per_cell, BLOCK),
        (per_cell, BLOCK.component_fanout()),
    ] {
        case.check(config);
    }
}

/// One contracted axis whose extent is no whole number of input vectors: the last vector's unused
/// components are never read, and the buffer is padded to whole vectors.
#[derive(Clone, Copy, Debug)]
struct Dot {
    k: usize,
    width: usize,
}

impl Dot {
    fn check(self, config: RegisterBlock) {
        let (oh, co) = (4, 4);
        let space = vec![(OH, oh), (CO, co), (CI, self.k)];
        let mut lhs = Operand::direct(&space, &[OH, CI], self.width, checked());
        lhs.shape[1] = self.k.next_multiple_of(self.width);
        Gather {
            tile: space.clone(),
            rhs: Operand::direct(&space, &[CI, CO], 2, BoundaryPolicy::Unchecked),
            lhs,
            out_axes: vec![OH, CO],
            space,
        }
        .check(config, &self);
    }
}

/// The vector walk's last, partial group of components.
#[test]
fn gather_contraction_tail_vector() {
    for (case, config) in [
        (Dot { k: 6, width: 4 }, BLOCK),
        (Dot { k: 7, width: 4 }, SPLIT),
        (Dot { k: 8, width: 4 }, BLOCK),
    ] {
        case.check(config);
    }
}
