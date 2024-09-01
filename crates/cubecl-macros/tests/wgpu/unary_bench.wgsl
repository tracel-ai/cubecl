@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<vec4<f32>>;

@group(0)
@binding(1)
var<storage, read_write> input_1_global: array<vec4<f32>>;

@group(0)
@binding(2)
var<storage, read_write> output_0_global: array<vec4<f32>>;

@group(0)
@binding(3)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let rank: u32 = info[0];
var l_0_0: u32;
var l_0_1: bool;
var l_0_2: bool;
var l_0_3: vec4<f32>;
var l_0_4: vec4<f32>;
l_0_0 = arrayLength(&output_0_global);
l_0_1 = id < l_0_0;
if l_0_1 {

for (var l_2_0: u32 = 0u; l_2_0 < 256u; l_2_0++) {
l_0_0 = l_2_0 % 2u;
l_0_2 = l_0_0 == 0u;
if l_0_2 {
l_0_3 = input_0_global[id];
l_0_4 = input_1_global[id];
l_0_3 = l_0_3 * l_0_4;
l_0_4 = cos(l_0_3);
l_0_3 = output_0_global[id];
l_0_3 = l_0_3 - l_0_4;
output_0_global[id] = vec4<f32>(l_0_3);
} else {
l_0_4 = input_0_global[id];
l_0_3 = input_1_global[id];
l_0_4 = l_0_4 * l_0_3;
l_0_4 = cos(l_0_4);
l_0_3 = output_0_global[id];
l_0_3 = l_0_3 + l_0_4;
output_0_global[id] = vec4<f32>(l_0_3);
}
}
}
}