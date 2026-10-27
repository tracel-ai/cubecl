//! The register nest behind [`Tile::reduce_axis`](crate::Tile::reduce_axis): reduce an input's
//! contracted axes into one accumulator cell at a time, under a [`Monoid`].

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::ops::contraction::resolve_nd_coords;
use crate::*;

#[cube]
pub(crate) fn register_data<Acc: Numeric, In: Numeric>(
    acc: &mut RegisterData,
    input: &Tile,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    // Drain the block the way it was built to reduce; the partials fit no other operator.
    comptime!(assert!(
        acc.accumulation.monoid() == monoid,
        "reduce: this accumulator sums under {:?} (stated at `Tile::accumulate`) but is being \
         reduced under {monoid:?}",
        acc.accumulation.monoid()
    ));

    let size!(V) = input.vector_size();
    register_data_body::<Acc, In, V>(acc, input, acc_space, monoid);
}

#[cube]
fn register_data_body<Acc: Numeric, In: Numeric, V: Size>(
    acc: &mut RegisterData,
    input: &Tile,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    let in_space = comptime!(input.place.space.clone());
    let vw = input.vector_size();
    let layout = comptime!(ReduceLayout::new(&in_space, &acc_space));
    let total_acc = comptime!(layout.total_acc);

    let count = comptime!(acc.mr * acc.nr);
    comptime!(assert!(
        total_acc == count * acc.vector_size,
        "reduce: RegisterData shape mismatch with accumulator space"
    ));
    let in_view = input.nd_packed::<In, V>(comptime!(Guard::Checked));
    let data = acc.data.at_mut::<Acc>();

    #[unroll]
    for a in 0..total_acc {
        let acc_coords =
            Coords::constant(comptime!(layout.acc_extents.clone())).unravel(comptime!(a as u32));

        let line_idx = comptime!(a / acc.vector_size);
        let component_idx = comptime!(a % acc.vector_size);
        let seed = data[line_idx].extract(comptime!(component_idx));

        let curr_val: Acc = element::<Acc, In, V>(
            &in_view,
            &in_space,
            &acc_space,
            &layout,
            &acc_coords,
            vw,
            seed,
            monoid,
        );

        let mut vec_line = data[line_idx];
        vec_line.insert(comptime!(component_idx), curr_val);
        data[line_idx] = vec_line;
    }
}

#[cube]
pub(crate) fn memory<Acc: Numeric, In: Numeric>(
    acc: &mut Memory,
    input: &Tile,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    let size!(V) = input.vector_size();
    memory_body::<Acc, In, V>(acc, input, acc_space, monoid);
}

#[cube]
fn memory_body<Acc: Numeric, In: Numeric, V: Size>(
    acc: &mut Memory,
    input: &Tile,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    let in_space = comptime!(input.place.space.clone());
    let vw = input.vector_size();
    let layout = comptime!(ReduceLayout::new(&in_space, &acc_space));
    let total_acc = comptime!(layout.total_acc);

    let ws = comptime!(acc.store.vector_size);
    // Preserve the native store width: retyping it as another vector width cannot lower on WGSL.
    let size!(W) = ws;
    comptime!(assert!(
        total_acc % ws == 0,
        "reduce: Memory total_acc must be divisible by store.vector_size"
    ));
    let total_lines = comptime!(total_acc / ws);
    let mut acc_view = acc.flat_accumulate::<Acc, W>(monoid);
    let in_view = input.nd_packed::<In, V>(comptime!(Guard::Checked));

    for line_idx in 0..total_lines {
        let seed_vec = acc_view.seed(line_idx);
        let mut result = seed_vec;

        #[unroll]
        for component_idx in 0..comptime!(ws) {
            let a = line_idx * comptime!(ws) + comptime!(component_idx);
            let acc_coords =
                Coords::constant(comptime!(layout.acc_extents.clone())).unravel(a.cast::<u32>());

            let seed = seed_vec.extract(comptime!(component_idx));
            let curr_val: Acc = element::<Acc, In, V>(
                &in_view,
                &in_space,
                &acc_space,
                &layout,
                &acc_coords,
                vw,
                seed,
                monoid,
            );

            result.insert(comptime!(component_idx), curr_val);
        }

        acc_view.commit(line_idx, result);
    }
}

/// Reduce `in_view` across the contracted axes into `seed` for the cell at `acc_coords`.
#[cube]
fn element<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &Masked<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: &Space,
    #[comptime] acc_space: &Space,
    #[comptime] layout: &ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] vw: usize,
    seed: Acc,
    #[comptime] monoid: Monoid,
) -> Acc {
    let contracted_per_step = comptime!(in_space.contracted_per_step(&layout.reduce_axes, vw));
    if comptime!(contracted_per_step > 1) {
        element_lines::<Acc, In, V>(
            in_view,
            in_space,
            acc_space,
            layout,
            acc_coords,
            contracted_per_step,
            seed,
            monoid,
        )
    } else {
        element_scalars::<Acc, In, V>(
            in_view, in_space, acc_space, layout, acc_coords, vw, seed, monoid,
        )
    }
}

/// [`element`]'s line path: one read serves `contracted_per_step` values.
#[cube]
#[allow(clippy::too_many_arguments)]
fn element_lines<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &Masked<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: &Space,
    #[comptime] acc_space: &Space,
    #[comptime] layout: &ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] contracted_per_step: usize,
    seed: Acc,
    #[comptime] monoid: Monoid,
) -> Acc {
    let mut acc_vec = Vector::<Acc, V>::cast_from(Monoid::identity::<Acc>(monoid));
    for p in 0..comptime!(layout.kc / contracted_per_step) {
        let reduce_coords = Coords::constant(comptime!(layout.reduce_extents.clone()))
            .unravel((p * comptime!(contracted_per_step)).cast::<u32>());

        let in_coords = resolve_nd_coords(
            in_space,
            acc_space,
            &layout.reduce_axes,
            acc_coords,
            &reduce_coords,
            contracted_per_step,
            true,
        );

        let in_vec = match comptime!(monoid) {
            Monoid::Sum => in_view.read(in_coords),
            Monoid::Prod | Monoid::Max | Monoid::Min => {
                let valid = in_view.is_in_bounds(in_coords.clone());
                select(
                    valid,
                    in_view.read(in_coords),
                    Vector::<In, V>::cast_from(Monoid::identity::<In>(monoid)),
                )
            }
        };
        acc_vec = monoid.combine::<Vector<Acc, V>>(acc_vec, Vector::<Acc, V>::cast_from(in_vec));
    }
    monoid.combine::<Acc>(
        seed,
        Monoid::reduce::<Acc, V>(acc_vec, contracted_per_step, monoid),
    )
}

/// [`element`]'s scalar path: one extract per contracted element.
#[cube]
#[allow(clippy::too_many_arguments)]
fn element_scalars<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &Masked<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: &Space,
    #[comptime] acc_space: &Space,
    #[comptime] layout: &ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] vw: usize,
    seed: Acc,
    #[comptime] monoid: Monoid,
) -> Acc {
    let mut curr_val = seed;
    let kc = comptime!(layout.kc);
    for p in 0..kc {
        let reduce_coords =
            Coords::constant(comptime!(layout.reduce_extents.clone())).unravel(p.cast::<u32>());

        let in_coords = resolve_nd_coords(
            in_space,
            acc_space,
            &layout.reduce_axes,
            acc_coords,
            &reduce_coords,
            vw,
            true,
        );

        // Memory reads return Sum's identity out of bounds; other monoids need it read in.
        let in_vec = match comptime!(monoid) {
            Monoid::Sum => in_view.read(in_coords),
            Monoid::Prod | Monoid::Max | Monoid::Min => {
                let valid = in_view.is_in_bounds(in_coords.clone());
                select(
                    valid,
                    in_view.read(in_coords),
                    Vector::<In, V>::cast_from(Monoid::identity::<In>(monoid)),
                )
            }
        };
        let in_val = if comptime!(vw <= 1) {
            in_vec.extract(0usize)
        } else {
            let in_component = resolve_reduce_component(
                in_space,
                acc_space,
                &layout.reduce_axes,
                acc_coords,
                &reduce_coords,
                vw,
            );
            in_vec.extract_dynamic(in_component)
        };
        let in_cast = Acc::cast_from(in_val);

        curr_val = monoid.combine::<Acc>(curr_val, in_cast);
    }

    curr_val
}

/// Comptime bookkeeping: the axes of `in_space` contracted against `acc_space`, and extents.
#[derive(Clone)]
struct ReduceLayout {
    reduce_axes: Vec<Axis>,
    reduce_extents: Vec<usize>,
    kc: usize,
    acc_extents: Vec<usize>,
    total_acc: usize,
}

impl ReduceLayout {
    fn new(in_space: &Space, acc_space: &Space) -> Self {
        let reduce_axes = Space::contracted(&[in_space], acc_space).to_vec();
        let reduce_extents = reduce_axes
            .iter()
            .map(|&a| in_space.extent(a))
            .collect::<Vec<usize>>();
        let kc = reduce_extents.iter().product::<usize>();
        let acc_extents = (0..acc_space.rank())
            .map(|p| acc_space.extent_at(p))
            .collect::<Vec<usize>>();
        let total_acc = acc_extents.iter().product::<usize>();
        Self {
            reduce_axes,
            reduce_extents,
            kc,
            acc_extents,
            total_acc,
        }
    }
}

/// The line component for the input's innermost axis, contracted or surviving.
#[cube]
// `#[cube]` converts each comptime argument with `From`, which gives no `&[Axis]` from a `&Vec`.
#[allow(clippy::ptr_arg)]
fn resolve_reduce_component(
    #[comptime] in_space: &Space,
    #[comptime] acc_space: &Space,
    #[comptime] reduce_axes: &Vec<Axis>,
    acc_coords: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] width: usize,
) -> usize {
    let in_rank = comptime!(in_space.rank());
    let fastest_axis = comptime!(in_space.axis_at(in_rank - 1));
    let raw_coord = if comptime!(acc_space.contains(fastest_axis)) {
        let pos = comptime!(acc_space.position(fastest_axis));
        acc_coords.at(comptime!(pos))
    } else {
        let pos = comptime!(reduce_axes.iter().position(|&r| r == fastest_axis).unwrap());
        reduce_coords.at(comptime!(pos))
    };
    raw_coord.remainder(comptime!(width as u32)).cast::<usize>()
}
