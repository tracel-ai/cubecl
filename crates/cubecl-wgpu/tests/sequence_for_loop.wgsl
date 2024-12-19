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
let l_0_0 = local_idx != 0u;
if l_0_0 {
return;
}
var l_mut_0_5: u32;
var l_mut_0_6: bool;
var l_mut_0_7: f32;
l_mut_0_5 = info[0u];
l_mut_0_6 = 0u < l_mut_0_5;
l_mut_0_7 = output_0_global[0u];
let l_0_1 = select(0f, l_mut_0_7, l_mut_0_6);
let l_0_2 = l_0_1 + 1f;
var l_mut_0_8: u32;
var l_mut_0_9: bool;
l_mut_0_8 = info[0u];
l_mut_0_9 = 0u < l_mut_0_8;
if l_mut_0_9 {
output_0_global[0u] = l_0_2;
}
var l_mut_0_10: u32;
var l_mut_0_11: bool;
var l_mut_0_12: f32;
l_mut_0_10 = info[0u];
l_mut_0_11 = 0u < l_mut_0_10;
l_mut_0_12 = output_0_global[0u];
let l_0_3 = select(0f, l_mut_0_12, l_mut_0_11);
let l_0_4 = l_0_3 + 4f;
var l_mut_0_13: u32;
var l_mut_0_14: bool;
l_mut_0_13 = info[0u];
l_mut_0_14 = 0u < l_mut_0_13;
if l_mut_0_14 {
output_0_global[0u] = l_0_4;
}
}