use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

macro_rules! impl_matmul_instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        /// All units in a plane do the same work in parallel
        ///
        /// Useful to mimic behaviour of CMMA when the instruction is unavailable
        /// Has lots of duplication and likely lots of bank conflicts
        pub struct $name<I: Numeric, O: Numeric, T: TmmConfig> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
            _config: PhantomData<T>,
        }

        #[cube]
        impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O, T> for $name<I, O, T> {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;

            type Lhs = OwnedTile<I>;
            type Rhs = OwnedTile<I>;
            type Out = OwnedTile<O>;

            fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
                execute::<I, O>(lhs, rhs, out);
            }

            fn init_lhs(#[comptime] config: T) -> Self::Lhs {
                OwnedTile::<I> {
                    handle: Array::<I>::new(Self::M * Self::K),
                    x: Self::M.runtime(),
                    y: Self::K.runtime(),
                    layout: as_dummy_layout(config.layout(Ident::Lhs)),
                }
            }

            fn init_rhs(#[comptime] config: T) -> Self::Rhs {
                OwnedTile::<I> {
                    handle: Array::<I>::new(Self::K * Self::N),
                    x: Self::K.runtime(),
                    y: Self::N.runtime(),
                    layout: as_dummy_layout(config.layout(Ident::Rhs)),
                }
            }

            fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs) {
                fill(slice, lhs);
            }

            fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs) {
                fill(slice, rhs)
            }

            fn init_output() -> Self::Out {
                let mut out = OwnedTile::<O> {
                    handle: Array::<O>::new(Self::M * Self::N),
                    x: Self::M.runtime(),
                    y: Self::N.runtime(),
                    layout: DummyLayout::RowMajor.runtime(),
                };

                for i in 0..Self::M * Self::N {
                    out.handle[i] = O::from_int(0);
                }

                out
            }

            fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>) {
                let line_size = Line::size(&slice[0]);

                for i in 0..out.x * out.y / line_size {
                    if comptime!(line_size == 1) {
                        slice[i] = Line::cast_from(out.handle[i]);
                    } else {
                        let mut line = Line::<C>::empty(line_size);
                        for j in 0..line_size {
                            line[j] = C::cast_from(out.handle[i * line_size + j]);
                        }
                        slice[i] = line;
                    }
                }
            }
        }

        impl<I: Numeric, O: Numeric, T: TmmConfig> Matmul<I, O> for $name<I, O, T> {
            type Config = T;

            fn check_config(_config: Self::Config) {}
        }
    };
}

impl_matmul_instruction!(DummyUnitInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(DummyUnitInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(DummyUnitInstruction8_32_16, 8, 32, 16);

#[cube]
fn fill<E: Numeric>(slice: &Slice<'_, Line<E>>, tile: &mut OwnedTile<E>) {
    let line_size = Line::size(&slice[0]);

    for i in 0..tile.x * tile.y / line_size {
        let line = slice[i];
        for j in 0..line_size {
            tile.handle[i * line_size + j] = line[j];
        }
    }
}

#[cube]
pub(crate) fn execute<I: Numeric, O: Numeric>(
    lhs: &OwnedTile<I>,
    rhs: &OwnedTile<I>,
    out: &mut OwnedTile<O>,
) {
    let m = lhs.x;
    let n = rhs.y;
    let k = rhs.x;

    for i in 0..m {
        for j in 0..n {
            for k_ in 0..k {
                let lhs_val = match comptime!(lhs.layout) {
                    DummyLayout::RowMajor => lhs.handle[i * k + k_],
                    DummyLayout::ColMajor => lhs.handle[k_ * m + i],
                };
                let rhs_val = match comptime!(rhs.layout) {
                    DummyLayout::RowMajor => rhs.handle[k_ * n + j],
                    DummyLayout::ColMajor => rhs.handle[j * k + k_],
                };

                out.handle[i * n + j] += O::cast_from(lhs_val * rhs_val);
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum DummyLayout {
    RowMajor,
    ColMajor,
}

impl CubeType for DummyLayout {
    type ExpandType = Self;
}

impl Init for DummyLayout {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for DummyLayout {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}

#[cube]
pub fn as_dummy_layout(#[comptime] layout: MatrixLayout) -> DummyLayout {
    match layout {
        MatrixLayout::RowMajor => DummyLayout::RowMajor,
        MatrixLayout::ColMajor => DummyLayout::ColMajor,
    }
}

#[derive(CubeType)]
pub struct OwnedTile<E: Numeric> {
    pub x: u32,
    pub y: u32,
    pub handle: Array<E>,
    pub layout: DummyLayout,
}
