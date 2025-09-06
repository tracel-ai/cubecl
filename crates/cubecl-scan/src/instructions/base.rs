use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait ScanInstruction: 'static + Send + Sync + std::fmt::Debug + CubeType {
    fn aggregate_line<N: Numeric>(line: Line<N>, #[comptime] line_size: u32) -> N;

    fn scan_line<N: Numeric>(
        base: N,
        line: Line<N>,
        #[comptime] line_size: u32,
        #[comptime] inclusive: bool,
    ) -> Line<N>;

    fn scan_plane<N: Numeric>(val: N, #[comptime] inclusive: bool) -> N;

    fn apply<N: Numeric>(a: N, b: N) -> N;
}
