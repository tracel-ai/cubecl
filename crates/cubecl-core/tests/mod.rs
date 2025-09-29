#[test]
#[cfg_attr(miri, ignore)]
fn compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/error/*.rs");
}
