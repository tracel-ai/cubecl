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
fn naming_kernel_f32_u8_bf16_i64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let _0 = info[1u];
let _1 = id < _0;
if _1 {
output_0_global[id] = 0f;
}
}