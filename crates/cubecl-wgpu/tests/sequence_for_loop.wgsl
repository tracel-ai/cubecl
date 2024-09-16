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
fn main(
    @builtin(local_invocation_index) local_idx: u32,
) {let rank: u32 = info[0];
var l_0_0: bool;
var l_0_1: f32;
l_0_0 = local_idx != 0u;
if l_0_0 {
return;
}
l_0_1 = output_0_global[0u];
l_0_1 = l_0_1 + 1f;
output_0_global[0u] = f32(l_0_1);
l_0_1 = output_0_global[0u];
l_0_1 = l_0_1 + 4f;
output_0_global[0u] = f32(l_0_1);
}