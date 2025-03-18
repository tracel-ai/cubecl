use cubecl::prelude::*;
use cubecl_core as cubecl;

// #[derive(CubeType)]
// pub enum CubeOption<T> {
//     Some(T),
//     None,
// }

// pub struct A;

// #[derive(Clone)]
// pub struct B;

// impl CubeType for A {
//     type ExpandType = B;
// }

// impl CubeDebug for B {}

// impl Init for B {
//     fn init(self, _: &mut Scope) -> Self {
//         self
//     }
// }

// #[derive(CubeType)]
// pub struct C {
//     x: u32,
// }

// #[derive(CubeType)]
// pub enum E {
//     First(u32),
//     Second,
// }

// #[derive(CubeType)]
// pub enum F<T: CubeType + IntoRuntime + Clone> {
//     First(T),
//     Second,
// }

// #[derive(CubeLaunch)]
// pub struct G {
//     x: u32,
// }

// #[derive(CubeLaunch)]
// pub struct H<
//     T: CubeType + IntoRuntime + Clone + LaunchArg + LaunchArgExpand + Send + Sync + 'static,
// > {
//     x: T,
// }

// #[derive(CubeLaunch)]
// pub enum I {
//     First(u32),
//     Second,
// }

// #[derive(CubeLaunch)]
// pub enum J<
//     T: CubeType
//         + IntoRuntime
//         + Clone
//         + LaunchArg
//         + LaunchArgExpand
//         + Send
//         + Sync
//         + 'static
//         + std::fmt::Debug
//         + Eq
//         + std::hash::Hash,
// > {
//     First(T),
//     Second,
// }

// #[derive(CubeType)]
pub struct D<T: CubeType + IntoRuntime> {
    x: T,
}

pub struct DExpand<T: CubeType + IntoRuntime> {
    x: <T as cubecl::prelude::CubeType>::ExpandType,
}
impl<T: CubeType + IntoRuntime> Clone for DExpand<T> {
    fn clone(&self) -> Self {
        Self { x: self.x.clone() }
    }
}
impl<T: CubeType + IntoRuntime> cubecl::prelude::CubeType for D<T> {
    type ExpandType = DExpand<T>;
}
impl<T: CubeType + IntoRuntime> cubecl::prelude::Init for DExpand<T> {
    fn init(self, context: &mut cubecl::prelude::Scope) -> Self {
        Self {
            x: cubecl::prelude::Init::init(self.x, context),
        }
    }
}
impl<T: CubeType + IntoRuntime> cubecl::prelude::CubeDebug for DExpand<T> {}
impl<T: CubeType + IntoRuntime> cubecl::prelude::IntoRuntime for D<T> {
    fn __expand_runtime_method(self, context: &mut cubecl::prelude::Scope) -> Self::ExpandType {
        let expand = DExpand {
            x: cubecl::prelude::IntoRuntime::__expand_runtime_method(self.x, context),
        };
        Init::init(expand, context)
    }
}
