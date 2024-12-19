@group(0)
@binding(0)
var<storage, read_write> output_0_global: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 4u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(4, 1, 1)
fn kernel_elect(
    @builtin(local_invocation_index) local_idx: u32,
) {
var l_mut_0_0: bool;
l_mut_0_0 = subgroupElect();
let l_0_1 = u32(l_mut_0_0);
var l_mut_0_2: u32;
var l_mut_0_3: bool;
l_mut_0_2 = info[0u];
l_mut_0_3 = local_idx < l_mut_0_2;
if l_mut_0_3 {
output_0_global[local_idx] = l_0_1;
}
}