use cubecl_core as cubecl;
use cubecl_core::cube;

mod common;

#[test]
pub fn kernel_compiles() {
    #[allow(unused)]
    #[cube]
    fn compiles() {
        let a = 1;
    }
}
