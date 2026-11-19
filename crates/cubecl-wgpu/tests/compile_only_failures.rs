//! A pass that discards its launches hears a kernel of the work that can
//! never compile; an autotune candidate that fails stays its race's own. A
//! binary of its own: the tests open a process-wide override.

use cubecl_server::runtime::Runtime;

type TestRuntime = cubecl_wgpu::WgpuRuntime;

cubecl_core::testgen_compile_only_failures!();
