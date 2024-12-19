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
let b_0_0 = local_idx != 0u;
if b_0_0 {
return;
}
var l_0_5: u32;
l_0_5 = info[0u];
let b_0_1 = select(f32(0), output_0_global[0u], 0u < l_0_5);
let b_0_2 = b_0_1 + 1f;
var l_0_6: u32;
var l_0_7: bool;
l_0_6 = info[0u];
l_0_7 = 0u < l_0_6;
if l_0_7 {
output_0_global[0u] = b_0_2;
}
var l_0_8: u32;
l_0_8 = info[0u];
let b_0_3 = select(f32(0), output_0_global[0u], 0u < l_0_8);
let b_0_4 = b_0_3 + 4f;
var l_0_9: u32;
var l_0_10: bool;
l_0_9 = info[0u];
l_0_10 = 0u < l_0_9;
if l_0_10 {
output_0_global[0u] = b_0_4;
}
}
