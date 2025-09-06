use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::instructions::ScanInstruction;

#[derive(Debug, CubeType, Clone)]
pub struct Add {}

#[cube]
impl ScanInstruction for Add {
    fn aggregate_line<N: Numeric>(line: Line<N>, #[comptime] line_size: u32) -> N {
        let mut sum = N::cast_from(0);
        #[unroll]
        for i in 0..line_size {
            sum += line[i];
        }
        sum
    }

    fn scan_line<N: Numeric>(
        mut base: N,
        line: Line<N>,
        #[comptime] line_size: u32,
        #[comptime] inclusive: bool,
    ) -> Line<N> {
        let mut res = Line::empty(line_size);

        #[unroll]
        for i in 0..line_size {
            if !inclusive {
                res[i] = base;
            }

            base += line[i];

            if inclusive {
                res[i] = base
            }
        }

        res
    }

    fn scan_plane<N: Numeric>(val: N, #[comptime] inclusive: bool) -> N {
        if inclusive {
            plane_inclusive_sum(val)
        } else {
            plane_exclusive_sum(val)
        }
    }

    fn apply<N: Numeric>(a: N, b: N) -> N {
        a + b
    }
}
