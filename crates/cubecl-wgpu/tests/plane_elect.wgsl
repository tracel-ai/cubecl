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
let _0 = subgroupElect();
let _1 = u32(_0);
var l_0_0: u32;
var l_0_1: bool;
l_0_0 = info[0u];
l_0_1 = local_idx < l_0_0;
if l_0_1 {
output_0_global[local_idx] = _1;
}
}
