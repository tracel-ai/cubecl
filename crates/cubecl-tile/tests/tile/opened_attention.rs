//! A destination carrying a state across the output tiles a routine hands it: a unit matmul
//! routine computing `q · kᵀ`, unchanged, and a destination running the online softmax on each
//! block of scores and contracting it with the values, so the routine and the attention after it
//! are one kernel.

use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Levels, Partitioning, Region, Semiring, SoftwareMatmul, Space, Tile, TileSpec, TileTensor,
    TileTensorLaunch,
    launch::{
        Destination, OpenedDestination, OpenedDestinationExpand, ServedDestination,
        ServedDestinationExpand,
    },
    ops::softmax::OnlineSoftmax,
    procedural::Procedural,
};

use super::online_softmax::{Attended, D, Q, S, V, check, inputs};
use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};

/// Every unit of every cube computing the output tiles its levels hand it, one at a time, each
/// opened on `O`'s tile and handed to `O` from the registers it was summed in.
#[cube(launch)]
fn unit_matmul<O: Destination>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &O::Arg<f32>,
    space: Partitioning,
    #[comptime] config: SoftwareMatmul,
) {
    let (a, b) = (lhs.tile(&space), rhs.tile(&space));
    let dest = O::serve::<f32, Const<1>>(out, &space);
    let c = dest.tile();
    for cube in &space {
        for unit in cube {
            let walk = unit.walk();
            let mut opened = dest.open(&unit, &c.at(&unit));
            for step in walk {
                let (a_s, b_s) = (a.at(&step), b.at(&step));
                let mut sum =
                    c.at(&step)
                        .register_accumulator::<f32>(&a_s, &b_s, config, Semiring::SUM_PROD);
                sum.mma(&a_s, &b_s);
                opened.consume(&step, &sum);
            }
            opened.close();
        }
    }
    dest.finish();
}

/// The mask's scalars, and the cells of a unit's rows of the output.
#[derive(CubeType, CubeLaunch)]
struct AttendingArg {
    keys: u32,
    queries: u32,
    scale: f32,
    #[cube(comptime)]
    causal: bool,
    #[cube(comptime)]
    cells: usize,
}

/// Each block of scores consumed by an online softmax and contracted with the values, the output
/// normalized after the last. Its tile is the scores' place: the bias over the queries and keys.
struct Attending;

/// A unit's rows of the output as they stand, their softmax state, and what each block reads.
#[derive(CubeType)]
struct Attention {
    acc: Tile,
    softmax: OnlineSoftmax<f32>,
    v: Tile,
    bias: Tile,
    out: Tile,
    scale: f32,
}

#[cube]
impl Destination for Attending {
    /// The values, the output and the mask.
    type Arg<E: Numeric> = (TileTensor<'static>, TileTensor<'static>, AttendingArg);
    type Served<E: Numeric> = AttendingServed;

    fn serve<E: Numeric, V: Size>(arg: &Self::Arg<E>, space: &Partitioning) -> AttendingServed {
        let bias = Procedural::new::<f32, Attended>(
            comptime!(space.space().subspace(&[Q, S])),
            Attended {
                keys: arg.2.keys,
                queries: arg.2.queries,
                causal: arg.2.causal,
            },
        )
        .tile_in::<f32>(space);
        AttendingServed {
            bias,
            v: arg.0.tile(space),
            out: arg.1.tile(space),
            scale: arg.2.scale,
            cells: comptime!(arg.2.cells),
        }
    }
}

/// [`Attending`] as served to the kernel: the bias, the values and the output, whole.
#[derive(CubeType)]
struct AttendingServed {
    bias: Tile,
    v: Tile,
    out: Tile,
    scale: f32,
    #[cube(comptime)]
    cells: usize,
}

#[cube]
impl ServedDestination for AttendingServed {
    type Opened = Attention;

    fn tile(&self) -> Tile {
        self.bias.clone()
    }

    fn open(&self, owner: &Region, sum: &Tile) -> Attention {
        let out = self.out.at(owner);
        let acc = out.register_accumulator::<f32>(
            &self.bias.at(owner),
            &self.v.at(owner),
            comptime!(SoftwareMatmul::new(self.cells)),
            Semiring::SUM_PROD,
        );
        Attention {
            acc,
            softmax: OnlineSoftmax::<f32>::along(sum, S),
            v: self.v.clone(),
            bias: self.bias.clone(),
            out,
            scale: self.scale,
        }
    }

    fn finish(&self) {}
}

#[cube]
impl OpenedDestination for Attention {
    fn consume(&mut self, at: &Region, sum: &Tile) {
        let correction = self.softmax.step(sum, &self.bias.at(at), self.scale);
        self.acc.along(V).mul(&correction);
        self.acc.mma(sum, &self.v.at(at));
    }

    fn close(&mut self) {
        self.acc.along(V).mul(&self.softmax.recip_l());
        self.acc.drained_into(&self.out);
    }
}

/// The attention of 24 queries over 40 keys, two rows a unit, four units a cube, a block of four
/// keys a step, through [`unit_matmul`] and [`Attending`], against the host.
fn attend(causal: bool) {
    let (queries, keys, head, value, rows, block) = (24, 40, 16, 8, 2, 4);
    let client: Client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            Levels::leaf(&[(Q, rows), (S, block), (D, head), (V, value)])
                .walk_every(&[S])
                .units_distributed(Q, 4)
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q, k, v) =
        inputs::<f32>(&client, queries, keys, keys, head, value, true);
    let out = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let arg = |handle: cubecl::std::tensor::TensorHandle, axes| {
        TileTensorLaunch::new(
            handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(axes),
        )
    };
    unit_matmul::launch::<Attending>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(q, &[Q, D]),
        // Stored transposed, so a step's rhs runs along the keys it scores.
        arg(k, &[D, S]),
        (
            arg(v, &[S, V]),
            arg(out.clone(), &[Q, V]),
            AttendingArgLaunch::new(
                keys as u32,
                queries as u32,
                1. / (head as f32).sqrt(),
                causal,
                rows * value,
            ),
        ),
        launcher.partitioning_arg(),
        SoftwareMatmul::new(rows * block),
    );
    let out = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    check(
        &out, &q_data, &k_data, &v_data, queries, keys, keys, head, value, causal, true,
    );
}

#[test]
fn a_destination_attends_across_the_tiles_of_a_routine() {
    attend(false);
}

#[test]
fn a_destination_attends_causally_across_the_tiles_of_a_routine() {
    attend(true);
}
