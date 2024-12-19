@group(0)
@binding(0)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 4u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(4, 1, 1)
fn kernel_sum(
    @builtin(local_invocation_index) local_idx: u32,
) {
var l_mut_0_1: f32;
var l_mut_0_3: u32;
var l_mut_0_4: bool;
var l_mut_0_5: f32;
l_mut_0_3 = info[0u];
l_mut_0_4 = local_idx < l_mut_0_3;
l_mut_0_5 = output_0_global[local_idx];
let l_0_0 = select(0f, l_mut_0_5, l_mut_0_4);
l_mut_0_1 = subgroupAdd(l_0_0);
let l_0_2 = local_idx == 0u;
if l_0_2 {
var l_mut_1_0: u32;
var l_mut_1_1: bool;
l_mut_1_0 = info[0u];
l_mut_1_1 = 0u < l_mut_1_0;
if l_mut_1_1 {
output_0_global[0u] = l_mut_0_1;
}
}
}