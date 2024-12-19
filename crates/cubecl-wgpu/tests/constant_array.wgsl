@group(0)
@binding(0)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> info: array<u32>;

const arrays_0: array<f32, 3> = array(f32(3u),f32(5u),f32(1u),);

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn constant_array_kernel_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let b_0_0 = info[1u];
let b_0_1 = id < b_0_0;
if b_0_1 {
let b_1_0 = arrays_0[id];
var l_1_1: u32;
var l_1_2: bool;
l_1_1 = info[0u];
l_1_2 = id < l_1_1;
if l_1_2 {
output_0_global[id] = b_1_0;
}
}
}