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
var l_0_3: u32;
l_0_3 = info[0u];
let b_0_0 = select(f32(0), output_0_global[local_idx], local_idx < l_0_3);
let b_0_1 = subgroupAdd(b_0_0);
let b_0_2 = local_idx == 0u;
if b_0_2 {
var l_1_0: u32;
var l_1_1: bool;
l_1_0 = info[0u];
l_1_1 = 0u < l_1_0;
if l_1_1 {
output_0_global[0u] = b_0_1;
}
}
}
