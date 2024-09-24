use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn line_simple<F: Float>(val: Array<Line<F>>) {
    val[0] + val[0];
}
