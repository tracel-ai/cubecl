use cubecl::ir::settings::Dim3;
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Cursor {
    slot: u32,
}

#[cube]
impl Cursor {
    fn plus(self, k: u32) -> u32 {
        self.slot + k
    }
}

#[cube]
fn method_argument(x: u32, #[comptime] result: Result<u32, ()>) -> comptime_type!(Result<u32, ()>) {
    let cursor = Cursor { slot: x };
    let _value = cursor.plus(result?);
    comptime!(Ok(7u32))
}

#[cube]
fn path_argument(x: u32, #[comptime] result: Result<u32, ()>) -> comptime_type!(Result<u32, ()>) {
    let cursor = Cursor { slot: x };
    let _value = Cursor::plus(cursor, result?);
    comptime!(Ok(7u32))
}

fn main() {
    for result in [Ok(3u32), Err(())] {
        let expected = result.map(|_| 7u32);

        let scope = Scope::root(KernelSettings::new(
            Dim3::new_single(),
            ExecutionMode::Checked,
            AddressType::U32,
        ));
        let x = 5u32.into_expand(&scope);
        assert_eq!(method_argument::expand(&scope, x, result), expected);

        let scope = Scope::root(KernelSettings::new(
            Dim3::new_single(),
            ExecutionMode::Checked,
            AddressType::U32,
        ));
        let x = 5u32.into_expand(&scope);
        assert_eq!(path_argument::expand(&scope, x, result), expected);
    }
}
