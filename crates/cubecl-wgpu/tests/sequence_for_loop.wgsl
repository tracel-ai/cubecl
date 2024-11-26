@group(0)
@binding(0)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn sequence_for_loop_kernel(
    @builtin(local_invocation_index) local_idx: u32,
) {
let _0 = local_idx != 0u;
if _0 {
return;
}
var l_0_0: u32;
l_0_0 = info[0u];
let _1 = select(f32(0), output_0_global[0u], 0u < l_0_0);
let _2 = _1 + 1f;
var l_0_1: u32;
var l_0_2: bool;
l_0_1 = info[0u];
l_0_2 = 0u < l_0_1;
if l_0_2 {
output_0_global[0u] = _2;
}
var l_0_3: u32;
l_0_3 = info[0u];
let _3 = select(f32(0), output_0_global[0u], 0u < l_0_3);
let _4 = _3 + 4f;
var l_0_4: u32;
var l_0_5: bool;
l_0_4 = info[0u];
l_0_5 = 0u < l_0_4;
if l_0_5 {
output_0_global[0u] = _4;
}
}
