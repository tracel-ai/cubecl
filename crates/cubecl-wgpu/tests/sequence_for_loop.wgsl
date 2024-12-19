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
var l_0_1: bool;
var l_0_2: f32;
l_0_0 = info[0u];
l_0_1 = 0u < l_0_0;
l_0_2 = output_0_global[0u];
let _1 = select(0f, l_0_2, l_0_1);
let _2 = _1 + 1f;
var l_0_3: u32;
var l_0_4: bool;
l_0_3 = info[0u];
l_0_4 = 0u < l_0_3;
if l_0_4 {
output_0_global[0u] = _2;
}
var l_0_5: u32;
var l_0_6: bool;
var l_0_7: f32;
l_0_5 = info[0u];
l_0_6 = 0u < l_0_5;
l_0_7 = output_0_global[0u];
let _3 = select(0f, l_0_7, l_0_6);
let _4 = _3 + 4f;
var l_0_8: u32;
var l_0_9: bool;
l_0_8 = info[0u];
l_0_9 = 0u < l_0_8;
if l_0_9 {
output_0_global[0u] = _4;
}
}
