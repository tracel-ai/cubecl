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
fn main(
    @builtin(local_invocation_index) local_idx: u32,
) {let rank: u32 = info[0];
let _0 = output_0_global[local_idx];
let _1 = subgroupAdd(_0);
let _2 = local_idx == 0u;
if _2 {
output_0_global[0u] = _1;
}
}