use cubecl_macros_2::cube2;

mod common;

#[test]
pub fn kernel_compiles() {
    #[allow(unused)]
    #[cube2]
    fn compiles() {
        let a = 1;
    }
}
