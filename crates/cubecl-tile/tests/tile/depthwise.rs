//! Depthwise convolution as a client of the tile DSL: the same gather-reduce [`conv`] runs, with
//! the channel axis *batched* rather than contracted.
//!
//! A dense convolution contracts input channels into output channels, so `CI` appears in the two
//! operands and not in the accumulator. Depthwise, each channel has its own filter and its own
//! output channel, so one axis `C` appears in *all three* operands. That is the whole difference.
//!
//! `C` is then a batch axis (an axis the walk splits and the leaf never folds), and the
//! contraction is over the window taps `RH`/`RW` alone.
//!
//! Stating it that way is what keeps this a space-and-projection change rather than a new kernel:
//! the input's [`Projection`] is the same `Ih = Oh*stride + Rh*dilation` gather, the weight drops
//! to `[RH, RW, C]`, and the accumulator keeps `C`. Nothing here re-derives a level hierarchy.
//!
//! Why it matters: `groups != 1` is refused outright by every accelerated convolution routine, so
//! a depthwise layer has exactly one implementation to fall back on. This path keeps channels
//! innermost, which a bandwidth-bound depthwise kernel needs: coalescing across `C` is the game.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{
    prelude::*,
    zspace::{Shape, shape},
};

use super::{Form, implied};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};

/// What runs on the cells the last level cuts out: a sixteen-scalar register block, no edge
/// specialization, no unit fan-out: the lines here run along the channel, not along `K`.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

// Output positions, window taps, and the one channel axis every operand shares.
const B: Axis = Axis(5);
const OH: Axis = Axis(0);
const OW: Axis = Axis(1);
const C: Axis = Axis(2);
const RH: Axis = Axis(3);
const RW: Axis = Axis(4);

/// The same body the dense convolution runs: this cube's box of the output, then this plane's
/// channel of it, contracted whole at the leaf. `C` being in the accumulator's own axes is what
/// makes it a batch axis rather than a contracted one; the leaf reads that off the spaces.
#[cube(launch)]
fn depthwise_kernel(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    for cube in space {
        let out = out.at(&cube);
        let input = input.at(&cube);
        let weight = weight.at(&cube);
        for plane in cube {
            for cell in plane {
                let mut out = out
                    .at(&cell)
                    .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
                out.mm(&input.at(&cell), &weight.at(&cell));
            }
        }
    }
}

/// Small integers, so the accumulation is exact in `f32` and the reference compares for equality.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
}

struct Depthwise {
    b: usize,
    oh: usize,
    ow: usize,
    c: usize,
    rh: usize,
    rw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    ph: usize,
    pw: usize,
}

impl Depthwise {
    fn in_h(&self) -> usize {
        (self.oh - 1) * self.sh + (self.rh - 1) * self.dh + 1 - 2 * self.ph
    }
    fn in_w(&self) -> usize {
        (self.ow - 1) * self.sw + (self.rw - 1) * self.dw + 1 - 2 * self.pw
    }

    /// One filter per channel, and no sum over input channels: that missing inner loop is the
    /// only way this differs from the dense reference.
    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let (ih, iw, c) = (self.in_h(), self.in_w(), self.c);
        let mut out = vec![0.0f32; self.b * self.oh * self.ow * c];
        for b in 0..self.b {
            for oh in 0..self.oh {
                for ow in 0..self.ow {
                    for ch in 0..c {
                        let mut acc = 0.0f32;
                        for rh in 0..self.rh {
                            for rw in 0..self.rw {
                                // Signed: the padded border reads outside the input and
                                // contributes nothing, which is what `checked` enforces.
                                let h = (oh * self.sh + rh * self.dh) as isize - self.ph as isize;
                                let w_ = (ow * self.sw + rw * self.dw) as isize - self.pw as isize;
                                if h < 0 || w_ < 0 || h >= ih as isize || w_ >= iw as isize {
                                    continue;
                                }
                                let x = input[((b * ih + h as usize) * iw + w_ as usize) * c + ch];
                                let k = weight[(rh * self.rw + rw) * c + ch];
                                acc += x * k;
                            }
                        }
                        out[((b * self.oh + oh) * self.ow + ow) * c + ch] = acc;
                    }
                }
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_ow: usize, tile_c: usize) {
        let launcher = implied(
            &cubecl::test_device().client(),
            Partitioning::new(
                Space::new(&[
                    (B, self.b),
                    (OH, self.oh),
                    (OW, self.ow),
                    (C, self.c),
                    (RH, self.rh),
                    (RW, self.rw),
                ]),
                Levels::leaf(&[(C, 1), (OW, 1), (OH, 1)])
                    .walk(&[(OW, tile_ow), (OH, tile_oh)])
                    .planes(&[(C, tile_c)])
                    .cubes(&[C, OW, OH])
                    .batches(&[B])
                    .build(),
            ),
            Form::Static,
        );

        // Two gathered physical axes, one per spatial pair; the channel axis rides identity, as
        // it does for the dense case: it is only the weight and accumulator that change.
        let in_spec = TileSpec::new(Projection::new(
            // The contracted taps come last: the gather leaf lines the input along the
            // fastest contracted axis, so `RW` has to be the operand's innermost logical
            // axis. The physical maps below stay in physical order (H, W, C) regardless.
            &[B, OH, OW, C, RH, RW],
            &[
                PhysicalAxisMap::of(B),
                PhysicalAxisMap::affine(&[(OH, self.sh), (RH, self.dh)])
                    .shifted(-(self.ph as isize)),
                PhysicalAxisMap::affine(&[(OW, self.sw), (RW, self.dw)])
                    .shifted(-(self.pw as isize)),
                PhysicalAxisMap::of(C),
            ],
        ))
        .boundary(BoundaryPolicy::Every(Boundary::Zero));

        let (got, want) = self.run(launcher, in_spec);
        for b in 0..self.b {
            for oh in 0..self.oh {
                for ow in 0..self.ow {
                    for ch in 0..self.c {
                        assert_eq!(
                            got.get_f32(&[b, oh, ow, ch]),
                            want[((b * self.oh + oh) * self.ow + ow) * self.c + ch],
                            "depthwise: wrong at ({b}, {oh}, {ow}, {ch})"
                        );
                    }
                }
            }
        }
    }

    fn run(&self, launcher: Launcher, in_spec: TileSpec) -> (HostData, Vec<f32>) {
        let client = cubecl::test_device().client();
        let f32_ty = f32::elem_type_native();

        let in_shape: Shape = shape![self.b, self.in_h(), self.in_w(), self.c];
        let w_shape: Shape = shape![self.rh, self.rw, self.c];
        let out_shape: Shape = shape![self.b, self.oh, self.ow, self.c];

        let in_data = ramp(in_shape.num_elements(), 7);
        let w_data = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(in_data.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(w_data.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), out_shape)
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        let w_spec = TileSpec::direct(&[RH, RW, C]);
        let out_spec = TileSpec::direct(&[B, OH, OW, C]);

        depthwise_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
            TileArgLaunch::new(w_handle.binding().into_tensor_arg(), f32_ty, 1, w_spec),
            TileArgLaunch::new(
                out_handle.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                out_spec,
            ),
            launcher.partitioning_arg(),
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference(&in_data, &w_data);
        (got, want)
    }
}

/// The shape a depthwise layer actually has: a 3x3 window, unit stride, many channels.
#[test]
fn depthwise_3x3_unit_stride() {
    Depthwise {
        b: 1,
        oh: 4,
        ow: 4,
        c: 8,
        rh: 3,
        rw: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        ph: 0,
        pw: 0,
    }
    .check(2, 2, 4);
}

/// Strided, which is how the encoder's stages downsample.
#[test]
fn depthwise_3x3_stride_2() {
    Depthwise {
        b: 1,
        oh: 3,
        ow: 3,
        c: 4,
        rh: 3,
        rw: 3,
        sh: 2,
        sw: 2,
        dh: 1,
        dw: 1,
        ph: 0,
        pw: 0,
    }
    .check(3, 3, 4);
}

/// A 5x5 window, which B4's later blocks use.
#[test]
fn depthwise_5x5() {
    Depthwise {
        b: 1,
        oh: 2,
        ow: 2,
        c: 4,
        rh: 5,
        rw: 5,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        ph: 0,
        pw: 0,
    }
    .check(2, 2, 4);
}

/// The shape the encoder is actually made of: batched, 3x3, unit stride, padded to keep the
/// resolution. This is the one `conv_direct` is the only candidate for today.
#[test]
fn depthwise_3x3_padded_batched() {
    Depthwise {
        b: 2,
        oh: 4,
        ow: 4,
        c: 8,
        rh: 3,
        rw: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        ph: 1,
        pw: 1,
    }
    .check(2, 2, 4);
}

/// Padded *and* strided, which is how a padded stage downsamples.
#[test]
fn depthwise_3x3_padded_stride_2() {
    Depthwise {
        b: 2,
        oh: 3,
        ow: 3,
        c: 4,
        rh: 3,
        rw: 3,
        sh: 2,
        sw: 2,
        dh: 1,
        dw: 1,
        ph: 1,
        pw: 1,
    }
    .check(3, 3, 4);
}
